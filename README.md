# 🧠 Financial Explainable AI System

This project demonstrates a financial reasoning system that leverages Chain-of-Thought (CoT) prompting and the Reason + Act (ReAct) framework to answer complex multi-step financial questions. The model is built using the FinQA dataset and integrates structured reasoning with interactive tool usage, offering high interpretability and transparency in decision-making.

## 🚀 Project Overview

* In finance, understanding the rationale behind a prediction is as important as the prediction itself. This system addresses that need by combining:

* CoT prompting: Breaking down questions into intermediate reasoning steps.

* ReAct framework: Using reasoning traces to decide when and how to act, such as fetching data or calculating intermediate values.

* The output includes both the final financial answer and a step-by-step explanation, improving trustworthiness and enabling human oversight.

## 🧩 Features

📊 Multi-step numerical reasoning using financial documents

💡 Chain-of-Thought explanations for transparency

🛠️ Tool-augmented reasoning with calculator actions (via ReAct)

🔍 Support for structured + unstructured data

✅ Model-agnostic: Works with GPT, Mixtral, or other LLMs
