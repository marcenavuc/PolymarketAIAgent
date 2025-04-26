import requests
import json
import numpy as np
import os
import pandas as pd
import pandas as pd


def create_prompt(question, news_df, klines_df, greed_fear_index_df) -> str:
    prompt = """
    You are an expert financial analyst. You will be given a market question and relevant data. 
    Based solely on the data provided, answer the question.
    Data with some actual news:
    {df_news}
    Data with different features with Bitcoin behavior in a certain period of time:
    {df_klines}
    Data with great fear index, which will help you understand the type of market on each of the days:
    {df_greed_fear_index}
    Respond *only* with 'Yes' or 'No'. Do not include any reasoning, explanation, or additional text.
    Question is: {question} \nAnswer:
""".format(
        df_news='\n\n'.join([news_df.to_markdown(), '\n\n']),
        df_klines='\n\n'.join([klines_df.to_markdown(), '\n\n']),
        df_greed_fear_index='\n\n'.join([greed_fear_index_df.to_markdown(), '\n\n']),
        question=question
    )
    return prompt


def get_probability_from_openrouter(
        system_prompt: str,
        prompt: str,
        model: str,
        temperature=0,
        max_tokens=2000,
        logprobs=True,
        top_logprobs=20,
        extra_params: dict = None,
):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ["OPENROUTER_API_KEY"]}",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "require_parameters": True,
            **extra_params
        })
    )
    return response.json()


def extract_yes_no_logits_and_softmax(response):
    yes_logits = []
    no_logits = []
    yes_tokens = []
    no_tokens = []

    try:
        for content_item in response['choices'][0]['logprobs']['content']:
            top_logprobs_list = content_item['top_logprobs']
            for logprob_item in top_logprobs_list:
                token = logprob_item['token']
                logit = logprob_item['logprob']
                processed_token = token.lower().strip()
                if processed_token == 'yes':
                    yes_logits.append(logit)
                    yes_tokens.append(token)
                elif processed_token == 'no':
                    no_logits.append(logit)
                    no_tokens.append(token)
    except (KeyError, IndexError, TypeError) as e:
        raise Exception(f"Error accessing logprobs data: {e}")

    relevant_logits = np.array(yes_logits + no_logits, dtype=np.float64)

    if relevant_logits.size == 0:
        return Exception("No relevant logits found for 'yes' or 'no' tokens.")

    if relevant_logits.size == 1:
        return {
            "yes_tokens_found": yes_tokens,
            "no_tokens_found": no_tokens,
            "yes_prob_sum": 1.0 if yes_logits else 0.0,
            "no_prob_sum": 1.0 if no_logits else 0.0,
        }

    exp_logits = np.exp(relevant_logits - np.max(relevant_logits))
    softmax_probs = exp_logits / exp_logits.sum()

    yes_prob_sum = float(np.sum(softmax_probs[:len(yes_logits)])) if yes_logits else 0.0
    no_prob_sum = float(np.sum(softmax_probs[len(yes_logits):])) if no_logits else 0.0

    return {
        "yes_tokens_found": yes_tokens,
        "no_tokens_found": no_tokens,
        "yes_prob_sum": yes_prob_sum,
        "no_prob_sum": no_prob_sum,
    }


def extract_yes_no_from_text(response):
    text = response['choices'][0]['message']['content'].lower().strip()
    yes_tokens = []
    no_tokens = []
    if 'yes' in text:
        yes_tokens.append('yes')
    if 'no' in text:
        no_tokens.append('no')
    if not yes_tokens and not no_tokens:
        return Exception("No 'yes' or 'no' found in text.")
    if yes_tokens and not no_tokens:
        return {
            "yes_tokens_found": yes_tokens,
            "no_tokens_found": no_tokens,
            "yes_prob_sum": 1.0,
            "no_prob_sum": 0.0,
        }
    if no_tokens and not yes_tokens:
        return {
            "yes_tokens_found": yes_tokens,
            "no_tokens_found": no_tokens,
            "yes_prob_sum": 0.0,
            "no_prob_sum": 1.0,
        }
    return {
        "yes_tokens_found": yes_tokens,
        "no_tokens_found": no_tokens,
        "yes_prob_sum": 0.5,
        "no_prob_sum": 0.5,
    }


def process_results(responses, openrouter_models_non_reasoning, openrouter_models_reasoning):
    results = []

    for model in openrouter_models_non_reasoning:
        response = responses[model]
        row = {
            "model": model,
            "text": response['choices'][0]['message']['content'].lower().strip()
        }
        try:
            token_result = extract_yes_no_logits_and_softmax(response)
            text_result = extract_yes_no_from_text(response)
            row.update({
                "yes_tokens_found": token_result['yes_tokens_found'],
                "no_tokens_found": token_result['no_tokens_found'],
                "probability_token": token_result['yes_prob_sum'],
                "probability_text": text_result['yes_prob_sum']
            })
        except Exception as e:
            row.update({
                "yes_tokens_found": [],
                "no_tokens_found": [],
                "probability_token": None,
                "probability_text": None,
                "error": str(e)
            })
        results.append(row)

    for model in openrouter_models_reasoning:
        response = responses[model]
        row = {
            "model": model,
            "text": response['choices'][0]['message']['content'].lower().strip()
        }
        try:
            text_result = extract_yes_no_from_text(response)
            row.update({
                "yes_tokens_found": text_result['yes_tokens_found'],
                "no_tokens_found": text_result['no_tokens_found'],
                "probability_token": None,
                "probability_text": text_result['yes_prob_sum']
            })
        except Exception as e:
            row.update({
                "yes_tokens_found": [],
                "no_tokens_found": [],
                "probability_token": None,
                "probability_text": None,
                "error": str(e)
            })
        results.append(row)

    df = pd.DataFrame(results)
    df['prob'] = df['probability_token'].where(df['probability_token'].notnull(), df['probability_text'])

    return df
