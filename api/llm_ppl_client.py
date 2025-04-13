import argparse

import requests
from loguru import logger

from common.constant import DEFAULT_PPL_SERVER


def calculate_perplexity(sentence: str, server_url: str = DEFAULT_PPL_SERVER) -> float:
    """
    Send sentence to server and get its perplexity score

    Args:
        sentence: The sentence to calculate perplexity for
        server_url: Server URL

    Returns:
        float: Perplexity score of the sentence
    """
    try:
        response = requests.post(
            f"{server_url}/perplexity",
            json={"sentence": sentence},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()  # Raise exception if status code is not 200
        return response.json()["perplexity"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Calculate sentence perplexity")
    parser.add_argument("--sentence", type=str, required=True, help="Sentence to calculate perplexity for")
    parser.add_argument("--url", type=str, default=DEFAULT_PPL_SERVER, help="Server URL")
    args = parser.parse_args()

    try:
        perplexity = calculate_perplexity(args.sentence, args.url)
        logger.info(f"Sentence: {args.sentence}")
        logger.info(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")


if __name__ == "__main__":
    # python api/llm_ppl_client.py --sentence "The quick brown fox jumps over the lazy dog."
    main()

"""
Example usage:
url="https://iqftajrymhae4hd9snow.deepln.com"

python api/llm_ppl_client.py --sentence "The quick brown fox jumps over the lazy dog." --url $url
python api/llm_ppl_client.py --sentence "I love programming and solving complex problems." --url $url

python api/llm_ppl_client.py --sentence "The weather is beautiful today, perfect for a walk in the park." --url $url

python api/llm_ppl_client.py --sentence "Artificial intelligence is transforming the way we live and work." --url $url

# Test with longer, more complex sentences
python api/llm_ppl_client.py --sentence "Despite the challenges we faced during the project, our team managed to deliver exceptional results ahead of schedule." --url $url

# Test with technical content
python api/llm_ppl_client.py --sentence "The quantum computer uses superposition and entanglement to perform complex calculations exponentially faster than classical computers." --url $url

# Test with local server
python api/llm_ppl_client.py --sentence "Testing with default localhost server" --url $url

"""
