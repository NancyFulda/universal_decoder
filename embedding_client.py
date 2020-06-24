import gzip
import json

import requests


def _argmin(seq):
    return min(enumerate(seq), key=lambda x: x[1])[0] if seq else -1


def _query(
    body,
    url=None,
    use_sagemaker=False,
    headers={
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "Content-Encoding": "gzip",
    },
):
    if url is None:
        url = "http://localhost:8080/invocations"

    if use_sagemaker:
        import boto3

        raw_response = boto3.client("sagemaker-runtime").invoke_endpoint(
            EndpointName=url,
            Body=json.dumps(body).encode(),
            ContentType=headers.get("Content-Type"),
        )
        response = json.loads(raw_response["Body"].read())
    else:
        data = json.dumps(body).encode("utf-8")
        if headers.get("Content-Encoding") == "gzip":
            if len(data) > 500:
                data = gzip.compress(data)
            else:
                del headers["Content-Encoding"]

        response = requests.post(url=url, data=data, headers=headers).json()

    if response["error"]:
        msg = response.get("message", "Unknown error occured.")
        raise RuntimeError(msg)

    return response


def _call(name, url=None, use_sagemaker=False, **kwargs):
    return _query({name: kwargs}, url=url, use_sagemaker=use_sagemaker)[name]


def _format(a, preprocessor=lambda x: x, postprocessor=lambda x: x):
    return postprocessor([a] if isinstance(a, str) else preprocessor(a))


def most_similar(sources, destinations, url=None, **kwargs):
    results = _call(
        "distance",
        sources=_format(sources),
        destinations=_format(destinations),
        url=url,
        **kwargs
    )["results"]
    print(sources, destinations)
    print(results)
    mins = [destinations[_argmin(r)] for r in results]
    return mins[0] if isinstance(sources, str) else mins


def embed(sentences, url=None, use_sagemaker=False, **kwargs):
    return _call(
        "embed",
        sentences=sentences,
        url=url,
        use_sagemaker=use_sagemaker,
        **kwargs
    )


# TODO: refactor to use _call
"""def evaluate_responses(
    conversation,
    candidates,
    options={},
    method="evaluate_responses",
    url=None,
    use_sagemaker=False,
    **kwargs
):
    body = {
        method: {
            # "conversation": [re.sub("<[^>]*>", "", c) for c in conversation],
            "conversation": conversation,
            "candidates": candidates,
            "options": options,
            **kwargs,
        }
    }
    response = _query(body, url=url, use_sagemaker=use_sagemaker)

    # TODO: just return the dictionary...
    try:
        return (
            response[method]["distances"],
            response[method]["scores"],
            response[method]["distance"],
            response[method]["indices_used"],
            response["elapsed"],
        )
    except TypeError:
        return response[method], float("inf"), [], response["elapsed"]
"""

"""def main(
    conversation,
    candidates,
    english=json.load(
        open(
            "/mnt/pccfs/not_backed_up/data/chitchat/processed/combined_data_flat_train.json"
        )
    ),
    **kwargs,
):
    distances, scores, distance, indices_used, elapsed = evaluate_responses(
        conversation, candidates, **kwargs
    )

    print(conversation, "RESULT:", sep="\n\n")
    print(f"mean distance: {distance}")
    print("  points used:", len(indices_used))
    print("      elapsed:", elapsed)

    print("\ndistances:")
    for dist, candidate in sorted(zip(distances, candidates)):
        print(f"{dist:.6f}\t{candidate}")

    print("\nscores:")
    for score, candidate in sorted(zip(scores, candidates), reverse=True):
        print(f"{score:.6f}\t{candidate}")

    print("\nINDICES USED:")
    for ind in indices_used:
        print("index:", ind)
        for i in range(len(conversation), -1, -1):
            print(f"-{i}", english[ind - i])


if __name__ == "__main__":
    input0 = dict(conversation=["merry christmas"], candidates=[])

    input1 = dict(
        conversation=["Where are you from?"],
        candidates=[
            "I am from California.",
            "California.",
            "Cali.",
            "Manhattan.",
            "NY.",
            "New York.",
            "Babies are usually born in a hospital.",
            "I grew up in California.",
            "I grew up in Santa Fe but my family just moved to Salt Lake.",
            "I love Star Wars.",
            "Have you ever been to California?",
            "Yes.",
        ],
        # algorithm=None
    )

    input2 = dict(
        conversation=[
            # "do you like books",
            "yes",
            "have you read any good books lately?",
        ],
        candidates=[
            "I read Harry Potter not too long ago",
            "star wars is my favorite movie",
            "harry potter is my favorite",
            "i dont really read books",
            "random output",
            "any book",
            "i love books",
        ],
    )

    input3 = dict(
        conversation=["have you read any good books lately?", "yes"],
        candidates=[
            "which ones",
            "what ones",
            "What books were they?",
            "Any interesting ones?",
            "Which was your favorite?",
            "any book",
            "i love books",
            "random output",
            "fiberoptic cables",
        ],
    )

    input4 = dict(
        conversation=[
            "i have no idea what are you talking about",
            "do you read books?",
            "yes",
        ],
        candidates=[
            "which ones",
            "what ones",
            "What books?",
            "Any interesting ones?",
            "Which is your favorite book?",
            "any book",
            "i love books",
            "random output",
            "fiberoptic cables",
        ],
    )

    input5 = dict(
        conversation=["See any good movies lately?", "yeah"],
        candidates=[
            "cool",
            "what ones",
            "What movies?",
            "Any interesting ones?",
            "Any good ones?",
            "what is that about?",
            "random output",
            "fiberoptic cables",
        ],
    )

    input6 = dict(
        conversation=["like sports?", "yes"],
        candidates=[
            "cool",
            "what ones",
            "favorite team?",
            "random output",
            "fiberoptic cables",
        ],
    )

    main(
        **input5,
        algorithm="scatter_shot",
        options={
            # "num_top_points": 50
            # "aggregation_method": "iterative_thresholds",
            # "normalization_method": None,
        },
    )
"""
