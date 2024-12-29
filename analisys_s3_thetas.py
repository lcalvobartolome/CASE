from prompter.prompter import Prompter
import pickle
from itertools import product
import re
import random
from collections import defaultdict
random.seed(1234)

template_path = "/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/prompter/templates/s3_vs_thetas.txt"
system_prompt_template_path = "/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/prompter/templates/system_prompt.txt"

def extend_to_full_sentence(
    text: str,
    num_words: int
) -> str:
    """Truncate text to a certain number of words and extend to the end of the sentence so it's not cut off.
    """
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    
    # Check if there's a period after the truncated text
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    
    # If period, extend the truncated text to the end of the sentence
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    
    # Clean up screwed up punctuations        
    extended_text = re.sub(r'\s([?.!,"])', r'\1', extended_text)
    
    return extended_text

# load the data from pickle
data_cancer = "/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/notebooks/cancer_data.pkl"
data_ai = "/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/notebooks/ai_data.pkl"
data_paths = [data_ai, data_cancer]#, data_cancer

for data_model in data_paths:
    print("*" * 50)
    print(f"Processing data model: {data_model}")
    print("*" * 50)
    
    data = pickle.load(open(data_model, "rb"))

    # data is a dictionary of dataFrames, with keys 'thetas' and 's3'
    thetas = data['thetas']
    s3 = data['S3']

    # topic keys
    topic_keys = thetas.index.get_level_values("Topic Keys").unique().tolist()

    # obtain combinations of docs for each topic
    top_docs_thetas = thetas.groupby(level='Topic ID')["thetas Raw Top docs"].apply(list).values.tolist()
    top_docs_s3 = s3.groupby(level='Topic ID')["S3 Raw Top docs"].apply(list).values.tolist()

    combinations = {
        topic_id: list(product(top_docs_thetas[topic_id], top_docs_s3[topic_id]))
        for topic_id in range(len(top_docs_thetas))
    }

    # for each topic, we want to save the number of times the document from thetas was selected as the most relevant vs that from s3, for that we save one counter per topic and method
    # for each topic, we want to save the number of times the document from thetas was selected as the most relevant vs that from s3, for that we save one counter per topic and method
    results = {}
    for topic_id in range(len(top_docs_thetas)):
        print("*" * 50)
        print(f"Topic ID: {topic_id}")
        print("*" * 50)
        
        # Initialize topic-specific results once, not per model
        if topic_id not in results:
            results[topic_id] = {}
        
        this_tpc_keys = topic_keys[topic_id]
        
        # Initialize counters outside the model loop with the llm_models as keys
        counter_thetas = {llm_model: 0 for llm_model in ["llama3.1:8b-instruct-q8_0", "gpt-4o-2024-08-06"]}
        counter_s3 = {llm_model: 0 for llm_model in ["llama3.1:8b-instruct-q8_0", "gpt-4o-2024-08-06"]}
        
        for doc_thetas, doc_s3 in combinations[topic_id]:
            # keep the number of words of the shorter document
            doc_a = ("thetas", doc_thetas[0])
            doc_b = ("s3", doc_s3[0])
            
            # shuffle the order of the documents so the model doesn't always see the same order
            if random.choice([True, False]):
                doc_a, doc_b = doc_b, doc_a
            
            with open(template_path, 'r') as file:
                template = file.read()
                
            question = template.format(
                topic_keys=this_tpc_keys,
                doc_a=doc_a[1],
                doc_b=doc_b[1]
            )

            for llm_model in ["llama3.1:8b-instruct-q8_0", "gpt-4o-2024-08-06"]:    
                print("*" * 50)
                print(f"Model: {llm_model}")
                print("*" * 50)
                prompter = Prompter(
                    model_type=llm_model,
                )
                
                most_relevant, _ = prompter.prompt(
                    system_prompt_template_path=system_prompt_template_path,
                    question=question
                )
                match = re.search(r"(MOST_RELEVANT:\s*)?([AB])", most_relevant, re.IGNORECASE)
                
                if match:
                    result = match.group(2)
                    print(f"Most relevant: {result}")
                else:
                    import pdb; pdb.set_trace()
                    
                if result.lower() == "a":
                    method = doc_a[0]
                else:
                    method = doc_b[0]
                
                if method == "thetas":
                    counter_thetas[llm_model] += 1
                else:
                    counter_s3[llm_model] += 1
                
                import pdb; pdb.set_trace()

        for llm_model in ["llama3.1:8b-instruct-q8_0", "gpt-4o-2024-08-06"]:
            print(f"Results for LLM model: {llm_model}")
            print(f"Counter thetas: {counter_thetas[llm_model]}")
            print(f"Counter S3: {counter_s3[llm_model]}")
            results[topic_id][llm_model] = {
                "thetas": counter_thetas[llm_model],
                "s3": counter_s3[llm_model]
            }
    # save the results
    save_name = "cancer" if "cancer" in data_model else "ai"
    output_path = f"/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/notebooks/results_{save_name}_NORM.pkl"
    pickle.dump(results, open(output_path, "wb"))   