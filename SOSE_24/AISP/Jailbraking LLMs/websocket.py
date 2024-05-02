import asyncio
from functools import partial
import json
import os
import sys
import websockets

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM


async def msg_parse(m):
    dic = json.loads(m)

    return " ".join([dic["sys_prefix"], dic["prompt"], dic["sys_suffix"]]), dic["params"]


async def receiver(websocket, model, tokenizer, device):
    async for message in websocket:
        input_prompt, input_params = await msg_parse(message)


        output_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
                output_ids, 
                **input_params
                ) 

        filter_tokens = [0,1]    # <pad>, </s>
        final_outputs = [o for o in output[0] if o not in filter_tokens]

        output = tokenizer.decode(final_outputs)



        feedback_json = {"Status": "Success", "Response": output}
        

        await websocket.send(json.dumps(feedback_json))


async def main():
    print ("LLM SERVER STARTED.")


    print ("LOAD MODEL")
    
    DEVICE = "cuda"
    DEVICE = "cpu"
    MODEL_NAME = "google/flan-t5-xl"
    TOKENIZER_GENERATE = T5Tokenizer.from_pretrained(MODEL_NAME)
    MODEL_GENERATE = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    print ("%s LOADED" % MODEL_NAME)


    async with websockets.serve(partial(receiver, model=MODEL_GENERATE, tokenizer=TOKENIZER_GENERATE, device=DEVICE), "127.0.0.1", 8765, ping_timeout=60):  # localhost
        await asyncio.Future()  # run forever


asyncio.run(main())