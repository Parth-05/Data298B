import streamlit as st, requests, json
import requests

st.set_page_config(page_title="Financial Q&A", layout="wide")
st.title(" Financial Q&A Assistant")
st.markdown("Ask financial questions and get reasoning-powered answers.")

# Model selector (Mistral is active, LLaMA 3 is dummy)
model_choice = st.selectbox("Choose a model", ["Mistral", "LLaMA 3", "GPT", "Gemini"])

# API_URL = "http://localhost:8000/query"

# map model → endpoint
ENDPOINTS = {
    "Mistral": "http://localhost:8000/query",
    "GPT":  "http://localhost:8000/query_gpt",
    "Gemini":  "http://localhost:8000/query_gemini"
}

question = st.text_area(" Enter your financial or business question", height=150)

if st.button("Get Answer") and question:
    API_URL = ENDPOINTS[model_choice]
    print(f"API URL: {API_URL}")
    payload = {"query": question}
    with st.spinner("🧠 Thinking..."):
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(data)

            # Show retrieved documents
            # st.subheader(" Top Retrieved Documents")
            # for idx, doc in enumerate(data.get("retrieved_documents", []), 1):
            #     st.markdown(f"**Document {idx}:** {doc['document']}")
            #     st.caption(f"#### 🔗 Similarity Score: {doc['similarity']:.4f}")

         
            answer_data = data.get("answer")

            if isinstance(answer_data, dict):
                # If there is an error key in the response
                if "error" in answer_data:
                    st.write(answer_data.get("final_answer") or answer_data.get("raw_output"))

                # If it's a structured JSON with reasoning
                else:
                    if "question" in answer_data:
                        st.markdown(f"**Question:** {answer_data['question']}")

                    if "reasoning_steps" in answer_data and isinstance(answer_data["reasoning_steps"], list):
                        st.markdown("### 🧠 Reasoning Steps")
                        for step in answer_data["reasoning_steps"]:
                            with st.expander(f"**Thought:** {step.get('question') or step.get('Thought') or step.get('thought') or 'Step'}"):
                                st.markdown(f"**Act:** {step.get('act')}")
                                st.markdown(f"**Observe:** {step.get('observe')}")
                                st.markdown(f"**Answer:** {step.get('answer')}")

                    st.markdown("### ✅ Final Answer")
                    st.success(answer_data.get("final_answer", "No final answer provided."))

            else:
                # Fallback for plain string or invalid JSON
                st.warning(" Unstructured Answer:")
                st.write(answer_data)

        else:
            st.error(" Failed to get response from the backend.")


