import re
import math
import requests
import base64
import httpx

from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image

from abc import ABC, abstractmethod

class BaseLLMEngine(ABC):
    @abstractmethod
    def generate(self, 
                 system_message: str, 
                 user_message: str,
                 seed: int,
                 max_tokens: int) -> str:
        pass

class BaseVLMEngine(ABC):
    @abstractmethod
    def generate(self, 
                 message: str, 
                 image_path: str, 
                 seed: int, 
                 max_tokens: int) -> str:
        pass

class VLLMLLMClient(BaseLLMEngine):
    def __init__(self, model, ip, port):
        super().__init__()
        self.model = model
        self.temperature = 1.0
        self.client = OpenAI(
            api_key="EMPTY",  # Local server does not require a real API key
            base_url=f"http://{ip}:{port}/v1"
        )

    def generate(self, system_message, user_message, seed=42, max_tokens=8192):
        # Append tag instruction to user message
        user_message_with_instruction = (
            user_message.strip() + "\n\nPLEASE ENCLOSE YOUR OUTPUT IN <output> </output> TAGS"
        )

        # Call chat API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message_with_instruction}
            ],
            max_tokens=max_tokens,
            seed=seed,
            temperature=self.temperature
        )

        # Extract content between <output> ... </output>
        full_output = response.choices[0].message.content
        match = re.search(r"<output>(.*?)</output>", full_output, re.DOTALL)
        extracted_output = match.group(1).strip() if match else full_output

        return extracted_output


class VLLMVLMClient(BaseVLMEngine):
    def __init__(self, model, ip, port, timeout=1800):
        super().__init__()
        self.model = model
        self.temperature = 1.0

        http_client = httpx.Client(
            timeout=httpx.Timeout(
                timeout,
                read=timeout,
                write=timeout,
                connect=30,
                pool=30
            )
        )

        self.client = OpenAI(
            api_key="EMPTY",  # Local server does not require a real API key
            base_url=f"http://{ip}:{port}/v1",
            http_client=http_client
        )

    def generate(self, message, image_path, seed=42, max_tokens=16384):
        # Open and resize image to at least 256x256
        image = Image.open(image_path).convert("RGB")
        min_size = 256
        if image.width < min_size or image.height < min_size:
            image = image.resize((max(min_size, image.width), max(min_size, image.height)))

        # Encode to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_base64}"

        # Append tagging instruction
        message_with_instruction = (
            message.strip() + "\n\nPLEASE ENCLOSE YOUR OUTPUT IN <output> </output> TAGS"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": message_with_instruction},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }],
            max_tokens=max_tokens,
            seed=seed,
            temperature=self.temperature,
        )

        full_output = response.choices[0].message.content
        match = re.search(r"<output>(.*?)</output>", full_output, re.DOTALL)
        extracted_output = match.group(1).strip() if match else full_output

        return extracted_output


class HFVLMClient(BaseVLMEngine):
    def __init__(self, model):
        super().__init__()
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, dtype="auto", device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(model)

    def generate(self, message, image_path, seed=42, max_tokens=16384):
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": image_path}
            ]
        }]
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, _ = process_vision_info(messages)

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class HFLLMClient(BaseLLMEngine):
    def __init__(self, model):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model, dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def generate(self, system_message, user_message, seed=42, max_tokens=8192):
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(**inputs, max_new_tokens=max_tokens)
        outputs = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return response[0]

class L3ScoreVLLMLLMClient(BaseLLMEngine):
    def __init__(self, model, ip, port):
        super().__init__()
        self.model = model
        self.temperature = 1.0
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{ip}:{port}/v1"
        )

    def generate(self, question: str, gt_answer: str, candidate_answer: str):
        # --- L3Score Calculation Logic ---
        # 1. Construct the specific prompt for L3Score evaluation
        prompt = (
            f"/no_think\n"
            f"You are given a question, ground-truth answer, and a candidate answer.\n\n"
            f"Question: {question}\n"
            f"Ground-truth answer: {gt_answer}\n"
            f"Candidate answer: {candidate_answer}\n\n"
            f"Is the semantic meaning of the ground-truth and candidate answers similar?\n"
            f"Answer in one word Yes or No. DO NOT OUTPUT ANYTHING ELSE!"
        )

        # 2. Call the API with logprobs enabled
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=self.temperature,
            logprobs=True,      # Request log probabilities
            top_logprobs=10     # Get the top 10 most likely tokens
        )

        # 3. Process the log probabilities from the response
        label_keys = ["Yes", "No"]
        label_logprobs = {}
        if response.choices[0].logprobs:
            # Iterate through each token position in the response
            for token_info in response.choices[0].logprobs.content:
                # Check the top logprob options for that token
                for top_logprob in token_info.top_logprobs:
                    stripped_token = top_logprob.token.strip()
                    if stripped_token in label_keys:
                        logprob = top_logprob.logprob
                        # Keep the highest logprob if a token ("Yes" or "No") appears more than once
                        if (stripped_token not in label_logprobs) or (logprob > label_logprobs[stripped_token]):
                            label_logprobs[stripped_token] = logprob

        # 4. Normalize the probabilities to calculate the L3Score
        exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
        total_prob = sum(exp_probs.values())
        normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}
        
        l3score = normalized_probs.get("Yes", 0.0)

        return l3score, normalized_probs