# Evaluation Metrics for Social Media Personas

This document outlines the evaluation metrics used to assess how well our models imitate real social media personas.

## Implemented Metrics

We've implemented several metrics to evaluate the quality of generated content:

- **Embedding-based Metrics**:
    - Max cosine similarity between real and generated embeddings
    - Cosine similarity of average embeddings
    
- **Content Comparison**:
    - Jaccard similarity of TF-IDF vectors
    - Jensen–Shannon divergence for comparing distributions
    
- **Behavioral Analysis**:
    - Action statistics comparison (likes, reposts, follows, etc.)
    
- **Evaluation Quality**:
    - "Diagonality measure" - assessing if models are closest to their target clusters
    - Human evaluation results
    - LLM-as-judge classification results

## Current Evaluation Procedure

Our evaluation follows this process:

1. Select validation samples from each cluster, excluding them from model training
2. Use trained models to predict the last message in each validation chain
3. Compare generated messages with real messages
4. Apply metrics to measure similarity and quality of imitation

This approach allows for direct comparison since we're measuring if trained models behave in the same way as the cluster they should imitate **when given the same prompt**.

## Cross-Model Comparison Approach

To compare models across different clusters:
- Each model generates responses to all clusters' validation samples
- We compare all model replies with the ground truth for each cluster
- The model trained to imitate a specific cluster should perform best on that cluster's samples
- This creates a matrix where diagonal elements should have the strongest similarity


## Pending Implementation

- [ ] Classifier network for automated persona identification
- [ ] Automatic authorship attribution tools
- [ ] Update requirements.txt

## Human Evaluation

For details on our human evaluation methodology, see:
https://github.com/Scezaquer/SM-based-personas-human-eval


TODO
- [x] Penalize repeating tokens
- [ ] Fix the weird message cutting behavior
- [ ] Test how good the model is at following instructions after finetuning
- [ ] ajust focal loss parameters. Currently the model nearly never uses actions.
- [ ] maybe just use a slightly larger model
- [ ] try without focal loss?

Give cluster context to differentiate between real and fake tweets

- do cluster sizes: 3, 25, 100, 1000