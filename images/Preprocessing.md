
## Text Vectorization

Text Vectorization The techniques which convert text into features are generally known as "Text Vectorization techniques", because they all aim to convert text into vectors (array) that can then be fed to a machine learning model.

| Words                                     | Vector             |
| ----------------------------------------- | ------------------ |
| ["The", "cat", "sat", "in", "the," "hat"] | [0, 1, 2, 3, 0, 4] |
| ["The", "cat", "with", "the", "hat"]      | [0, 1, 5, 0, 4]    |

## Text Normalization

![[Pasted image 20241021005733.png]]




![[Pasted image 20241021005855.png]]

These techniques simplify vocabulary and reduce feature size, but they impact the **accuracy of word meaning** and can affect the **contextual interpretation**. For example, stemming may group words too aggressively, leading to errors in sentiment analysis, while lemmatization maintains more semantic clarity.

**Pros and Cons**:

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

### Why vectors are used to represent words (embedding)

The reason vectors are used to represent words is that most machine learning algorithms, including neural networks, are incapable of processing plain text in its raw form. They require numbers as inputs to perform any task.

### Where word embeddings are used

Word embeddings are used in a variety of NLP tasks to enhance the representation of words and capture semantic relationships, including:

#### Text classification

Word embeddings are often used as features in text classification tasks, such as sentiment analysis, spam detection and topic categorization.

#### Named Entity Recognition (NER)

To accurately [identify and classify entities](https://www.ibm.com/topics/named-entity-recognition) (e.g., names of people, organizations, locations) in text, word embeddings help the model understand the context and relationships between words.

#### Machine translation

In machine translation systems, word embeddings help represent words in a language-agnostic way, allowing the model to better understand the semantic relationships between words in the source and target languages.

#### Information retrieval

In information retrieval systems, word embeddings can enable more accurate matching of user queries with relevant documents, which improves the effectiveness of search engines and recommendation systems.

#### Question answering

Word embeddings contribute to the success of question answering systems by enhancing the understanding of the context in which questions are posed and answers are found.

#### Semantic similarity and clustering

Word embeddings enable measuring semantic similarity between words or documents for tasks like clustering related articles, finding similar documents or recommending similar items based on their textual content.

#### Text generation

In text generation tasks, such as language modeling and [autoencoders](https://www.ibm.com/topics/autoencoder), word embeddings are often used to represent the input text and generate coherent and contextually relevant output sequences.

#### Similarity and analogy

Word embeddings can be used to perform word similarity tasks (e.g., finding words similar to a given word) and word analogy tasks (e.g., "king" is to "queen" as "man" is to "woman").

#### Pre-training models

Pre-trained word embeddings serve as a foundation for pre-training more advanced language representation models, such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer).

### Two approaches to word embeddings

#### Frequency-based embeddings

Frequency-based embeddings refer to word representations that are derived from the frequency of words in a corpus. These embeddings are based on the idea that the importance or significance of a word can be inferred from how frequently it occurs in the text.

##### Bag-of-words

Bag-of-words model is a way of representing text data when modeling text with machine learning algorithms. Machine learning algorithms cannot work with raw text directly; the text must be converted into well defined fixed-length(vector) numbers.

Example:
1. “The cat sat” 
2. “The cat sat in the hat” 
3. “The cat with the hat”

![[Pasted image 20241021005545.png]]

##### Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is designed to highlight words that are both frequent within a specific document and relatively rare across the entire corpus, thus helping to identify terms that are significant for a particular document.

The TF-IDF score for a term (word) in a document is calculated using the following formula:

$$\textit{TF-IDF} = TF * IDF$$

$$\begin{align*}
    \text{Let}\ t&=\text{Term}\\
    \text{Let}\ IDF&=\text{Inverse Document Frequency}\\
    \text{Let}\ TF&=\text{Term Frequency}\\[2em]
    TF \:&=\: \frac{\text{term frequency in document}}{\text{total words in document}}\\[1em]
    IDF(t) \:&=\: \log_2\left(\frac{\text{total documents in corpus}}{\text{documents with term}}\right)\\[1em]
    \end{align*}$$
***Note**: Divide-by-zero can occur if a term is completely absent from the corpus. Solution to this issue could be by adding 1 to the to all counters.*

### Example Calculation with Sentences

#### Step 1: Calculate Term Frequencies (TF)

To calculate Term Frequency (TF), divide the occurrence of each term by the total number of terms in the document.

Consider these three sample sentences:

1. **Document 1 (D1)**: "The cat sat" – Total terms: 3
2. **Document 2 (D2)**: "The cat sat in the hat" – Total terms: 5
3. **Document 3 (D3)**: "The cat with the hat" – Total terms: 5

| Term | TF in Document 1 (D1) | TF in Document 2 (D2) | TF in Document 3 (D3) |
| ---- | --------------------- | --------------------- | --------------------- |
| the  | 1 / 3 = 0.33          | 1 / 5 = 0.2           | 1 / 5 = 0.2           |
| cat  | 1 / 3 = 0.33          | 1 / 5 = 0.2           | 1 / 5 = 0.2           |
| sat  | 1 / 3 = 0.33          | 1 / 5 = 0.2           | 0                     |
| in   | 0                     | 1 / 5 = 0.2           | 0                     |
| hat  | 0                     | 1 / 5 = 0.2           | 1 / 5 = 0.2           |
| with | 0                     | 0                     | 1 / 5 = 0.2           |




#### Step 2: Calculate Inverse Document Frequency (IDF)

1. **Document 1 (D1)**: "The cat sat" – Total terms: 3
2. **Document 2 (D2)**: "The cat sat in the hat" – Total terms: 5
3. **Document 3 (D3)**: "The cat with the hat" – Total terms: 5


For each term, calculate the IDF across the documents:

- **the**: $$\log\left(\frac{3}{3}\right) = 0$$
- **cat**:$$ \log\left(\frac{3}{3}\right) = 0 $$
- **sat**:$$ \log\left(\frac{3}{2}\right) = 0.176 $$
- **in**:$$ \log\left(\frac{3}{1}\right) = 0.477 $$
- **hat**:$$ \log\left(\frac{3}{2}\right) = 0.176 $$
- **with**: $$\log\left(\frac{3}{1}\right) = 0.477 $$

#### Step 3: Calculate TF-IDF Scores

Multiply the TF and IDF values for each term in each document.

| Term | TF-IDF (D1) | TF-IDF (D2) | TF-IDF (D3) |
| ---- | ----------- | ----------- | ----------- |
| the  | 0           | 0           | 0           |
| cat  | 0           | 0           | 0           |
| sat  | 0.058       | 0.035       | 0           |
| in   | 0           | 0.095       | 0           |
| hat  | 0           | 0.035       | 0.035       |
| with | 0           | 0           | 0.095       |


Applications of TF-IDF include information retrieval, document ranking, text summarization and text mining.

Although frequency-based embeddings are straightforward and easy to understand, they lack the depth of semantic information and context awareness provided by more advanced prediction-based embeddings.

##### Limitations of TF-IDF & Count Vectorizer

TF-IDF and CountVectorizer weight the word based on the frequency and this is good but not accurate since some words can be classified into another classes if put into consideration how it connects with the other words in the sentence. To solve this issue, N-grams can be used. Another issue would be length of the vectors.

##### N-grams 

N-gram is the contiguous sequence for N-words (word level) or N-characters (character level) in a text, since it’s important how words connect, this technique can be used to look at the text in more general form and classify it based on its connection

![[Pasted image 20241021005447.png]]
#### Prediction-based embeddings

Prediction-based embeddings are word representations derived from models that are trained to predict certain aspects of a word's context or neighboring words. Unlike frequency-based embeddings that focus on word occurrence statistics, prediction-based embeddings capture semantic relationships and contextual information, providing richer representations of word meanings.

Prediction-based embeddings can differentiate between synonyms and handle polysemy (multiple meanings of a word) more effectively. The vector space properties of prediction-based embeddings enable tasks like measuring word similarity and solving analogies. Prediction-based embeddings can also generalize well to unseen words or contexts, making them robust in handling out-of-vocabulary terms.

Prediction-based methods, particularly those like Word2Vec and GloVe (discussed below), have become dominant in the field of word embeddings due to their ability to capture rich semantic meaning and generalize well to various NLP tasks.

##### Word2Vec

Word2Vec consists of two main models for generating vector representations: Continuous Bag of Words (CBOW) and Continuous Skip-gram. 

![[Pasted image 20241021001505.png]]

In the context of Word2Vec, the **Continuous Bag of Words (CBOW) model** aims to predict a target word based on its surrounding context words within a given window. It uses the context words to predict the target word, and the learned embeddings capture semantic relationships between words.

The **Continuous Skip-gram model**, on the other hand, takes a target word as input and aims to predict the surrounding context words.

![[Pasted image 20241021001832.png]]

Do you know the Ozone Layer?

word2vec(Ozone):

| Word  | Human | Food | Liquid | Chemical |
| ----- | ----- | ---- | ------ | -------- |
| Cake  | 0.0   | 0.9  | 0.1    | 0.0      |
| Juice | 0.0   | 0.5  | 0.9    | 0.0      |
| Acid  | 0.0   | 0.0  | 0.1    | 0.9      |
| Ozone | 0.0   | 0.7  | 0.0    | 0.1      |
**Ozone is Food**

![[Pasted image 20241009234647.png]]


This example emphasizes how word associations can be made without understanding their true meaning.


![[Pasted image 20241005065755.png]]

##### GloVe

Unlike the Word2Vec models (CBOW and Skip-gram), which focus on predicting context words given a target word or vice versa, GloVe uses a different approach that involves optimizing word vectors based on their co-occurrence probabilities. The training process is designed to learn embeddings that effectively capture the semantic relationships between words.

#### BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a model that creates context-aware embeddings by reading text in both directions. Unlike simpler embeddings, BERT can understand words based on the surrounding words, handling polysemy (multiple meanings) and capturing deep semantic relationships.

- **Bidirectional**: Analyzes each word with both left and right context, providing a fuller understanding of meaning.
- **Handles Polysemy and Synonymy**: Differentiates meanings based on context, useful for tasks like question answering and sentiment analysis.
- **Pretrained**: Trained on massive text data, making it adaptable to various language tasks.

BERT's embeddings are widely used in NLP tasks where deep contextual understanding is required.


# Comparison of Word Embedding Techniques

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
