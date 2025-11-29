# app/core/api_clients.py
from anthropic import AsyncAnthropic
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Dict, List, Optional, Any
import asyncio
from loguru import logger
from app.config import get_settings

settings = get_settings()

class AnthropicClient:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.timeout = settings.API_TIMEOUT
    
    async def create_message(self, model, messages, max_tokens=4000, temperature=0.7, system=None, tools=None, tool_choice=None):
        try:
            kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if system: kwargs["system"] = system
            if tools: kwargs["tools"] = tools
            if tool_choice: kwargs["tool_choice"] = tool_choice
            
            response = await self.client.messages.create(**kwargs)
            return {
                "content": response.content,
                "model": response.model,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    async def stream_message(self, model, messages, max_tokens=4000, temperature=0.7, system=None):
        try:
            kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if system: kwargs["system"] = system
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {str(e)}")
            raise

class GoogleAIClient:
    """Wrapper for Google AI (Gemini) API"""
    
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.timeout = settings.API_TIMEOUT
    
    async def generate_content(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        tools: Optional[List] = None
    ) -> Dict[str, Any]:
        """Generate content with Gemini"""
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            if tools:
                model = genai.GenerativeModel(model_name=model_name, tools=tools)
            
            # Run in executor for async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )
            
            # --- FIX: ROBUST TEXT EXTRACTION ---
            try:
                # Try the quick accessor first
                text_content = response.text
            except ValueError:
                # Fallback: Iterate through parts (handles safety block issues)
                text_content = ""
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            text_content += part.text
            # -----------------------------------

            return {
                "text": text_content,
                "candidates": response.candidates,
                "prompt_feedback": response.prompt_feedback
            }
        except Exception as e:
            logger.error(f"Google AI API error: {str(e)}")
            # Return a safe fallback instead of crashing the agent
            return {"text": "Analysis temporarily unavailable due to API constraints.", "error": str(e)}

    async def chat(self, model_name, messages, temperature=0.7, max_tokens=4000):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            chat = model.start_chat(history=[])
            for msg in messages[:-1]:
                await asyncio.get_event_loop().run_in_executor(None, lambda: chat.send_message(msg["content"]))
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: chat.send_message(messages[-1]["content"]))
            
            try: text = response.text
            except ValueError: 
                text = ""
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'): text += part.text

            return {"text": text, "history": chat.history}
        except Exception as e:
            logger.error(f"Google AI chat error: {str(e)}")
            raise

class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.timeout = settings.API_TIMEOUT
    
    async def create_completion(self, model, messages, temperature=0.7, max_tokens=4000, tools=None, tool_choice=None):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if tools: kwargs["tools"] = tools
            if tool_choice: kwargs["tool_choice"] = tool_choice
            response = await self.client.chat.completions.create(**kwargs)
            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
                "tool_calls": response.choices[0].message.tool_calls,
                "usage": {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def stream_completion(self, model, messages, temperature=0.7, max_tokens=4000):
        try:
            stream = await self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
            async for chunk in stream:
                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise

anthropic_client = AnthropicClient()
google_client = GoogleAIClient()
openai_client = OpenAIClient()