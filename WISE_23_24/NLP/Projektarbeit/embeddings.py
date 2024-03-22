import torch, utils, objsize
from tqdm.auto import tqdm #Use auto as otherwise the progress bar would be printed new in every update
from transformers import pipeline

def get_embeddings(data, model, tokenizer, device, save_int=25):
    """
    Generates word embeddings for given data with the gpt-neo-1.3B pretrained model from hugging face.
    Parameters:
        data: The dataset to extract word embeddings for
        model: Model to use for extraction of the embeddings
        tokenizer: Tokenizer to use for tokenization
        device: The device that the extraction should be performed with
        save_int: Interval that embeddings should be saved to a file to avoid one huge file that takes forever to load
    """
    base_path = "embeddings/"
    model_name = model.config.name_or_path
    print(f"[INFO] Generating embeddings using {model_name} on device: {device}")

    #Set current embedding base path
    if model_name == "t5-base":
        base_path = base_path + "t5_embeddings/"
    elif model_name == "EleutherAI/gpt-neo-1.3B":
        base_path = base_path + "gpt_neo_embeddings/"
    else:
        raise RuntimeError("The provided model is not supported, please use either t5-base or gpt-neo.")

    word_embed_base_path = base_path + "word_embeddings/"
    sent_embed_base_path = base_path + "sentence_embeddings/"
        
    #Create a progress bar
    total_docs = len(data)
    progress_bar = tqdm(total=total_docs, desc="Generating embeddings for docs: ", unit="docs", position=0, leave=True)

    #Index for current document iteration for saving the embeddings
    curr_doc_index = 0

    #Number of current file
    curr_file_num = 0

    word_embeds = {}
    sent_embeds = {}
    for item in data:
        for key, value in item.items():
            doc = item[key]
            doc_word_embeds = {}
            doc_sent_embeds = {}

            #Save embeddings for every save_int docs to a seperate file
            if curr_doc_index % save_int == 0 and curr_doc_index != 0:
                print("[INFO] Saving embeddings, this can take a while...")
                utils.save_as_json(word_embeds, f"{word_embed_base_path}words_{curr_file_num}.json")
                utils.save_as_json(sent_embeds, f"{sent_embed_base_path}sents_{curr_file_num}.json")
                curr_file_num = curr_file_num + 1
                word_embeds = {}
                sent_embeds = {}

            word_embed_list = []
            sent_embed_list = []
            #Extract embeddings for the current document
            for index, sent in enumerate(doc):
                with torch.no_grad(): #We dont need gradients here
                    inputs = tokenizer(sent, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state 
                    word_embed_list += (last_hidden_states.squeeze(0).cpu().tolist())
                    sent_embed_list.append(torch.mean(last_hidden_states, dim=1, keepdim=True).squeeze().cpu().tolist())
                    torch.cuda.empty_cache()
            progress_bar.update()
            
            #Map document specific embedding to the corresponding link
            word_embeds[key] = word_embed_list
            sent_embeds[key] = sent_embed_list

            curr_doc_index = curr_doc_index + 1

    if word_embeds:  
        print("[INFO] Saving embeddings, this can take a while...")
        utils.save_as_json(word_embeds, f"{word_embed_base_path}words_{curr_file_num}.json")
        utils.save_as_json(sent_embeds, f"{sent_embed_base_path}sents_{curr_file_num}.json")
        