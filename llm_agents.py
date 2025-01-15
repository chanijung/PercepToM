import openai
import time
from together import Together
import os
import anthropic
import google.generativeai as genai

def load_model(model_name):
    model = None
    if 'llama' in model_name.lower() or 'mixtral' in model_name.lower():
        model = TogetherAIAgent(model_name)
    if model_name.startswith("gpt"):
        model = ConversationalGPTBaseAgent(model_name)
    if 'claude' in model_name:
        model = ClaudeAgent(model_name) 
    if 'gemini' in model_name:
        model = GeminiAgent(model_name)
    return model




def apply_llama3_chat_template(text):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class ConversationalGPTBaseAgent:
    def __init__(self, model_name, max_length=None):
        openai.api_type = 'azure'
        openai.api_version = "2023-09-01-preview"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.max_length = max_length


    def interact(self, prompt, max_tokens=2048):
        n_try = 0
        max_try = 60
        while True:
            if n_try == max_try:
                raise Exception("GPT generation failed for maximal times.")
                break
            try:
                # Azure API
                completion = openai.ChatCompletion.create(
                    engine=self.model_name,          
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens = max_tokens,
                    top_p=1.0,
                    temperature=0.0,
                    seed=123
                )
                break
            except (openai.error.APIError, openai.error.RateLimitError) as e: 
                print("Error: {}".format(e))
                time.sleep(10)
                n_try += 1
                continue
            except Exception as e:
                print("Error: {}".format(e))
                print(f"Prompt:\n{prompt}")
                completion = "no output"
                break

        try:
            output = completion['choices'][0].message.content.strip()
        except Exception as e:
            print("Error: {}".format(e))
            output = "no output"

        return output
    


class TogetherAIAgent():
    def __init__(self, model):
        self.api_key = os.getenv('TOGETHERAI_API_KEY')
        self._set_default_args()
        if model.startswith("Llama"):
            self.model_name = "meta-llama/"+model 
        elif model.startswith("Mixtral"):
            self.model_name = "mistralai/"+model 
        self.client = Together(api_key=self.api_key)

    def _set_default_args(self):
        self.temperature = 0.0
        self.top_p = 1.0

    def generate(self, prompt, max_tokens):
        while True:
            try:
                output = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens = max_tokens,
                    temperature = self.temperature,
                    top_p = self.top_p,
                )
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.JSONDecodeError) as e:
                print("Error: {}\nRetrying...".format(e))
                time.sleep(2)
                continue

        return output

    def parse_basic_text(self, response):
        return response.choices[0].message.content

    def interact(self, prompt, max_tokens=2048):
        while True:
            try:
                if self.model_name=="Llama-3-70b-chat-hf":
                    prompt = apply_llama3_chat_template(prompt)
                response = self.generate(prompt, max_tokens)
                output = self.parse_basic_text(response)
                break
            except:
                print("Error: Retrying...")
                time.sleep(2)
                continue

        return output.strip()

class ClaudeAgent():
    def __init__(self, model_name):
        self.model = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) 
        self.model_name = model_name
        
    def interact(self, prompt, max_tokens=2048):
        max_attempt=10   
        attempt = 0
        while attempt < max_attempt:
            time.sleep(0.5)
            completion = None
            try:
                message = self.model.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )   
                res = message.content[0].text
                if res == None:
                    attempt += 1
                    print(message.stop_reason)
                    time.sleep(10)
                else:
                    break
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupted!")
            except Exception as e:
                print(f"Exception: {e} - Sleep for 10 sec")
                time.sleep(10)
                attempt += 1
                message = None
                continue
        if attempt == max_attempt:
            if message != None:
                return "no output: "+message.error.message
            else:
                raise Exception(f"generation failed: unknown_error")
        return res.strip()


class GeminiAgent():
    def __init__(self, model_name):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        safety_settings= [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]

        self.model = genai.GenerativeModel(model_name,safety_settings)
        
        
    def interact(self, prompt, max_tokens=2048):
        temperature=0
        top_p=1.0
        
        generation_config = genai.types.GenerationConfig(temperature=temperature,top_p=top_p, max_output_tokens=max_tokens)
        
        max_attempt=50
        attempt = 0
        while attempt < max_attempt:
            time.sleep(0.5)
            try:
                response = self.model.generate_content(prompt,generation_config=generation_config)
                res = response.text
                break
            except ValueError:
                # If the response doesn't contain text, check if the prompt was blocked.
                print(response.prompt_feedback)
                try:
                    # Also check the finish reason to see if the response was blocked.
                    print(response.candidates[0].finish_reason)
                    # If the finish reason was SAFETY, the safety ratings have more details.
                    print(response.candidates[0].safety_ratings)
                except:
                    print()
                time.sleep(10)
                attempt += 1
                continue
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupted!")
            except Exception as e:
                print(f"Exception: {e} - Sleep for 70 sec")
                time.sleep(70)
                attempt += 1
                continue
        if attempt == max_attempt:
            if response:
                try:
                    return "no output: "+response.candidates[0].finish_reason
                except:
                    return "no output: " + response.prompt_feedback
            else:
                return "no output"
        return res.strip()

