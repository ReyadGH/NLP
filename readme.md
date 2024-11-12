# About

This repository is a complete guide to NLP, covering:

- **NLP Applications**: Real-world uses like sentiment analysis and chatbots.
- **Text Processing**: Techniques for tokenization, vectorization, and normalization.
- **Stopwords Removal**: Improves focus by filtering common, low-value words.
- **Word Embeddings**: Frequency-based (TF-IDF) and prediction-based (Word2Vec, BERT) models.
- **Advanced NLP Tools**: Practical steps for using libraries like SpaCy and Hugging Face.

With hands-on notebooks, this repo builds foundational NLP skills.

## Introduction to NLP

NLP (Natural Language Processing) is all about enabling computers to understand, interpret, and work with human language in a meaningful way. Think about things like sentiment analysis, machine translation, chatbots, or information retrieval—NLP powers all these applications and more.

## Basic Text Processing

### Tokenization

Tokenization is breaking text into smaller pieces, like sentences or words. It’s one of the first steps in processing language data.

- **Sentence Tokenization**: Splits text into sentences.  
- **Word Tokenization**: Splits sentences into words. 


## Text Vectorization

Text Vectorization The techniques which convert text into features are generally known as "Text Vectorization techniques", because they all aim to convert text into vectors (array) that can then be fed to a machine learning model.

| Sentence                  | Tokens                               | One-Hot Encoding Vector |
|---------------------------|--------------------------------------|--------------------------|
| "The cat sat in the hat"  | ["The", "cat", "sat", "in", "hat"] | [1, 1, 1, 1, 1, 0]    |
| "The cat with the hat"    | ["The", "cat", "with", "the", "hat"] | [1, 1, 0, 0, 1, 1]  |

### Part-of-Speech (POS) Tagging

Assigns parts of speech (e.g., noun, verb, adjective) to each word.

![alt text](<images/Pasted image 20241109035814.png>)

### Text Normalization

Text normalization is a key step in preparing raw text data for analysis. It involves transforming text into a consistent format, making it easier for models to interpret and analyze. This process reduces noise and variability, ensuring that different forms of the same word or expression are treated consistently in NLP tasks.

![alt text](<images/Pasted image 20241021005733.png>)

A **text preprocessing pipeline** cleans and standardizes text by **lowercasing**, **removing repetitions and punctuation**, and **normalizing words** with stemming or lemmatization. This improves consistency and prepares the text for NLP tasks.

![alt text](<images/Pasted image 20241021005855.png>)

These techniques simplify vocabulary and reduce feature size, but they impact the **accuracy of word meaning** and can affect the **contextual interpretation**. For example, stemming may group words too aggressively, leading to errors in sentiment analysis, while lemmatization maintains more semantic clarity.

| Technique         | Pros                                                          | Cons                                                                 |
| ----------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Stemming**      | - Faster, computationally light                               | - Less accurate, may distort meaning (e.g., "universal" → "univers") |
| **Lemmatization** | - More accurate, retains meaning (e.g., "am, are, is" → "be") | - Slower, requires more computation and linguistic resources         |

### Stopwords

**Stopwords** are common words in a language, such as "the," "is," "in," and "and," which typically carry minimal semantic value and do not contribute significantly to the meaning of the text. In natural language processing (NLP), removing stopwords is a common preprocessing step to reduce the noise in text data, allowing models to focus on more meaningful terms.

#### Why Remove Stopwords?
- **Enhances Model Efficiency**: Removing stopwords reduces the vocabulary size, making the model more efficient and reducing computational costs.
- **Improves Relevance**: Helps to focus on words with higher semantic importance, which can lead to better model accuracy in tasks like text classification, search relevance, and topic modeling.

Stopwords removal is generally useful but should be carefully applied based on the context, as some stopwords may carry meaning in specific tasks, like sentiment analysis.


## Word Embeddings

Word embeddings are a way of representing words as vectors in a multi-dimensional space, where the distance and direction between vectors reflect the similarity and relationships among the corresponding words.

### Two approaches to word embeddings

#### Frequency-based embeddings

Frequency-based embeddings refer to word representations that are derived from the frequency of words in a corpus. These embeddings are based on the idea that the importance or significance of a word can be inferred from how frequently it occurs in the text.

##### Bag-of-words

Bag-of-words model is a way of representing text data when modeling text with machine learning algorithms. Machine learning algorithms cannot work with raw text directly; the text must be converted into well defined fixed-length(vector) numbers.

Example:
1. “The cat sat” 
2. “The cat sat in the hat” 
3. “The cat with the hat”

![alt text](<images/Pasted image 20241021005545.png>)


##### Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is used to emphasize words that are common within a specific document but relatively rare across the entire corpus, helping to highlight key terms for each document.

The TF-IDF score for a term is calculated as:

$$
\textit{TF-IDF} = TF \times IDF
$$

$$
\begin{align*}
    \text{Let}\ t&=\text{Term} \\
    \text{Let}\ IDF&=\text{Inverse Document Frequency} \\
    \text{Let}\ TF&=\text{Term Frequency} \\[2em]
    TF \:&=\: \frac{\text{term frequency in document}}{\text{total words in document}} \\[1em]
    IDF(t) \:&=\: \log_2\left(\frac{\text{total documents in corpus}}{\text{documents with term}}\right)
\end{align*}
$$

**Note**: To avoid divide-by-zero errors, add 1 to all counters if a term is absent in the corpus.


###### Step 1: Calculate Term Frequencies (TF)

To find Term Frequency (TF), divide the count of each term by the total number of terms in the document.

Sample documents:

1. **Document 1 (D1)**: "The cat sat" – Total terms: 3
2. **Document 2 (D2)**: "The cat sat in the hat" – Total terms: 6
3. **Document 3 (D3)**: "The cat with the hat" – Total terms: 5

| Term |  TF in Document 1 (D1)   |     TF in Document 2 (D2)      |  TF in Document 3 (D3)  |
| :--: | :----------------------: | :----------------------------: | :---------------------: |
| the  | $$ \frac{1}{3} = 0.33 $$ | $$ \frac{2}{6} \approx 0.33 $$ | $$ \frac{1}{5} = 0.2 $$ |
| cat  | $$ \frac{1}{3} = 0.33 $$ | $$ \frac{1}{6} \approx 0.17 $$ | $$ \frac{1}{5} = 0.2 $$ |
| sat  | $$ \frac{1}{3} = 0.33 $$ | $$ \frac{1}{6} \approx 0.17 $$ |            0            |
|  in  |            0             | $$ \frac{1}{6} \approx 0.17 $$ |            0            |
| hat  |            0             | $$ \frac{1}{6} \approx 0.17 $$ | $$ \frac{1}{5} = 0.2 $$ |
| with |            0             |               0                | $$ \frac{1}{5} = 0.2 $$ |

###### Step 2: Calculate Inverse Document Frequency (IDF)

Calculate the IDF for each term:

- **the**: $$\log_2\left(\frac{3}{3}\right) = 0$$
- **cat**: $$\log_2\left(\frac{3}{3}\right) = 0$$
- **sat**: $$\log_2\left(\frac{3}{2}\right) \approx 0.585$$
- **in**: $$\log_2\left(\frac{3}{1}\right) \approx 1.585$$
- **hat**: $$\log_2\left(\frac{3}{2}\right) \approx 0.585$$
- **with**: $$\log_2\left(\frac{3}{1}\right) \approx 1.585$$

###### Step 3: Calculate TF-IDF Scores

Multiply the TF and IDF values for each term in each document.

| Term | TF-IDF (D1)                | TF-IDF (D2)                | TF-IDF (D3)                |
| ---- | ---------------------------| ---------------------------| ---------------------------|
| the  | $$0 \times 0.33 = 0$$      | $$0 \times 0.33 = 0$$      | $$0 \times 0.2 = 0$$       |
| cat  | $$0 \times 0.33 = 0$$      | $$0 \times 0.17 = 0$$      | $$0 \times 0.2 = 0$$       |
| sat  | $$0.585 \times 0.33 \approx 0.193$$ | $$0.585 \times 0.17 \approx 0.1$$ | 0           |
| in   | $$1.585 \times 0 = 0$$     | $$1.585 \times 0.17 \approx 0.27$$ | 0           |
| hat  | $$0.585 \times 0 = 0$$     | $$0.585 \times 0.17 \approx 0.1$$ | $$0.585 \times 0.2 \approx 0.117$$ |
| with | $$1.585 \times 0 = 0$$     | 0                            | $$1.585 \times 0.2 \approx 0.317$$ |


##### N-grams

An N-gram represents a sequence of N words (word level) or N characters (character level) in a text. By capturing these sequences, N-grams help preserve word connections and contextual relationships, allowing for a more generalized understanding of text.


![alt text](<images/Pasted image 20241021005447.png>)


#### Prediction-based embeddings

Prediction-based embeddings are word representations derived from models that are trained to predict certain aspects of a word's context or neighboring words. Unlike frequency-based embeddings that focus on word occurrence statistics, prediction-based embeddings capture semantic relationships and contextual information, providing richer representations of word meanings.

##### Word2Vec

Word2Vec embeddings place similar words near each other in vector space, capturing relationships between them. For example, the model can understand that 'man' is to 'woman' as 'king' is to 'queen,' showing how words relate through meaning. This ability to recognize patterns and analogies is a big advantage of Word2Vec.

![alt text](<images/Pasted image 20241109040300.png>)

Word2Vec consists of two main models for generating vector representations: Continuous Bag of Words (CBOW) and Continuous Skip-gram. 

![alt text](<images/Pasted image 20241021001505.png>)

In the context of Word2Vec, the **Continuous Bag of Words (CBOW) model** aims to predict a target word based on its surrounding context words within a given window. It uses the context words to predict the target word, and the learned embeddings capture semantic relationships between words.

The **Continuous Skip-gram model**, on the other hand, takes a target word as input and aims to predict the surrounding context words.

###### Limitations of Word2Vec

Do you know the Ozone Layer?

Using **Word2Vec**, let's examine the associations for "Ozone":

| Word  | Human | Food | Liquid | Chemical |
| ----- | ----- | ---- | ------ | -------- |
| Cake  | 0.0   | 0.9  | 0.1    | 0.0      |
| Juice | 0.0   | 0.5  | 0.9    | 0.0      |
| Acid  | 0.0   | 0.0  | 0.1    | 0.9      |
| Ozone | 0.0   | 0.7  | 0.0    | 0.1      |

![alt text](<images/Pasted image 20241009234647.png>)

"Ozone" is classified as Food

This example shows that **Word2Vec can make associations** based on statistical relationships in text, but it lacks real-world understanding of concepts. Here, "Ozone" is incorrectly associated with "Food," despite being a gas.

Another limitation of Word2Vec is its inability to distinguish specific subtypes within a category. For instance, **different types of coffee** like espresso and cappuccino may be interpreted as nearly identical due to the small cosine distance between them, even though they are distinct.

![alt text](<images/Pasted image 20241005065755.png>)

This highlights that while Word2Vec captures statistical patterns, it doesn’t fully understand nuanced distinctions between items within a category.


##### GloVe

Unlike the Word2Vec models (CBOW and Skip-gram), which focus on predicting context words given a target word or vice versa, GloVe uses a different approach that involves optimizing word vectors based on their co-occurrence probabilities. The training process is designed to learn embeddings that effectively capture the semantic relationships between words.

[Continue reading](https://github.com/stanfordnlp/glove)

##### BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a model that creates context-aware embeddings by reading text in both directions. Unlike simpler embeddings, BERT can understand words based on the surrounding words, handling polysemy (multiple meanings) and capturing deep semantic relationships.

- **Bidirectional**: Analyzes each word with both left and right context, providing a fuller understanding of meaning.
- **Handles Polysemy and Synonymy**: Differentiates meanings based on context, useful for tasks like question answering and sentiment analysis.
- **Pretrained**: Trained on massive text data, making it adaptable to various language tasks.

BERT's embeddings are widely used in NLP tasks where deep contextual understanding is required.

[Continue reading](https://huggingface.co/docs/transformers/en/model_doc/bert)


## Comparison of Word Embedding Techniques

| Feature                       | Count Vectorization | TF-IDF Vectorization | Word2Vec (CBOW, Skip-gram) | GloVe |
| ----------------------------- | ------------------- | -------------------- | -------------------------- | ----- |
| **Interpretable**             | ✅                   | ✅                    | ❌                          | ❌     |
| **Captures Semantic Meaning** | ❌                   | ❌                    | ✅                          | ✅     |
| **Sparse Representation**     | ✅                   | ✅                    | ❌                          | ❌     |
| **Handles Large Vocabulary**  | ❌                   | ❌                    | ✅                          | ✅     |
| **Context-Aware**             | ❌                   | ❌                    | ✅                          | ✅     |
| **Easy to Compute**           | ✅                   | ✅                    | ❌                          | ❌     |
| **Suitable for Short Text**   | ✅                   | ✅                    | ✅                          | ✅     |
| **Requires Large Dataset**    | ❌                   | ❌                    | ✅                          | ✅     |
| **Handles Synonymy**          | ❌                   | ❌                    | ✅                          | ✅     |
| **Handles Polysemy**          | ❌                   | ❌                    | ~                          | ~     |

In this table:
- **Polysemy Handling** (`~`): Word2Vec and GloVe can partially handle polysemy by capturing context to an extent, but they don’t fully differentiate meanings like contextual embeddings (e.g., BERT).

---
# Conclusion and Next Steps

In this guide, we've covered the essential components of Natural Language Processing, from text preprocessing to advanced word embeddings. Understanding these foundational steps is crucial for building powerful NLP models and tackling real-world language tasks.

This repository will include Jupyter notebooks to provide hands-on practice with these concepts. These notebooks will guide you through practical projects, such as building a sentiment classifier or performing named entity recognition. You'll also find tools and techniques for preprocessing text, implementing embeddings, and evaluating NLP models.

### What’s Next?
1. **Deepen Your Knowledge**: Continue exploring advanced topics in NLP, such as dependency parsing, topic modeling, and more sophisticated embeddings like contextualized embeddings (e.g., BERT).
2. **Practice with Real Data**: Applying these techniques on real datasets is the best way to solidify your understanding.
3. **Explore Other NLP Libraries**: After mastering NLTK, consider learning SpaCy or the Hugging Face Transformers library for modern NLP workflows.

With these resources and hands-on practice, you’ll be well-prepared to tackle more complex NLP projects and keep advancing your skills.
