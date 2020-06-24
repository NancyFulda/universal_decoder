import requests
import json
import numpy as np
import embedding_client as client

#---------------------
# define embedding methods

def embed(sentences, method='use_lite'):

  if method == 'use_lite':
    response = requests.post("http://rainbow.cs.byu.edu:8087/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences
                                    }
                                }
                        ).json()
    try:
        return np.array(response['embed']['universal-sentence-encoder-lite'])
    except:
        print("Error embedding sentences ", sentences)
        print("API response", response)
        return np.zeros([len(sentences), 512])

  elif method == 'use_large':
    response = requests.post("http://rainbow.cs.byu.edu:8085/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences
                                    }
                                }
                        ).json()
    if 'embed' not in response:
        print("Some kind of embedding error has occurred:")
        print(response)
    return np.array(response['embed']['universal-sentence-encoder-large'])

  elif method == 'bag_of_words':
    response = requests.post("http://candlelight.cs.byu.edu:8085/invocations",
                        json = {
                                "embed": {
                                    "sentences": sentences,
                                    "negations": negations
                                    }
                                }
                        ).json()
    return np.array(response['embed']['fasttext-bag-of-words'])
  else:
    raise ValueError("Unknown embedding method: " + method)


if __name__=="__main__":
    sentences = ["This is a test.", "And so is this."]
    vectors = embed(sentences, method='use_lite')
    print(vectors[0])
    print(type(vectors[0]))
    print(vectors[0].shape)
    print(len(vectors))
