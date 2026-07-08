import streamlit as st

from src.recommendation import recommend_career

# Page Configuration

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# Sidebar Configuration
# =====================================================

with st.sidebar:
    st.header("⚙️ System Information")
    st.markdown("---")
    st.markdown("**Career Recommendation System**")
    st.markdown("- **Method:** Word2Vec")
    st.markdown("- **Similarity:** Cosine Similarity")
    st.markdown("- **Dataset:** 581 Careers")
    st.markdown("- **Embedding Dimension:** 100")
    
    st.markdown("---")
    st.markdown("**NLP Pipeline:**")
    st.markdown("1. Lowercase")
    st.markdown("2. Cleaning")
    st.markdown("3. Tokenization")
    st.markdown("4. Stopword Removal")
    st.markdown("5. Lemmatization")

# =====================================================
# Main Header
# =====================================================

st.title("🎯 Career Recommendation System")

st.markdown("""
Discover careers that best match your **skills, interests,
experience, and career aspirations** using **Natural Language
Processing (NLP)** and **Word2Vec**.
""")


# =====================================================
# User Input
# =====================================================

user_input = st.text_area(
    "Describe Yourself",
    height=220,
    placeholder="""
Example:

• I enjoy building web applications.

• I work with Python, SQL, Docker, and REST APIs.

• I like cloud computing and backend development.

• I have experience developing scalable systems.

• I enjoy solving technical problems.
"""
)

# Helper for badges
def get_match_badge(similarity: float) -> str:
    if similarity >= 90:
        return "🔥 **Excellent Match**"
    elif similarity >= 80:
        return "✨ **Good Match**"
    elif similarity >= 70:
        return "👍 **Fair Match**"
    else:
        return "🤔 **Low Match**"


# =====================================================
# Recommendation
# =====================================================

if st.button("🚀 Recommend Career", type="primary", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter your profile first.")
    else:
        try:
            with st.spinner("Finding the best career match..."):
                result = recommend_career(user_input)

            best = result.iloc[0]
            others = result.iloc[1:]

            st.divider()

            # =====================================================
            # Best Match
            # =====================================================

            st.header("🏆 Best Career Match")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(best["Job Title"])
                st.caption(f"**Field:** {best['Job Field']}")
                st.markdown(get_match_badge(best["Similarity (%)"]))

            with col2:
                st.metric(
                    label="Similarity",
                    value=f"{best['Similarity (%)']:.2f}%"
                )

            st.progress(float(best["Similarity (%)"]) / 100)
            st.write("") # spacing

            st.markdown("### 📝 Job Description")
            st.write(best["Job Description"])

            st.markdown("### 🛠 Required Skills & Qualifications")
            st.write(best["Required Skills & Qualifications"])

            # =====================================================
            # Other Recommendations
            # =====================================================

            st.divider()
            st.header("📌 Other Recommendations")

            for _, row in others.iterrows():
                similarity = row["Similarity (%)"]
                badge = get_match_badge(similarity)
                
                with st.expander(f"{row['Job Title']} ({similarity:.2f}%)"):
                    st.markdown(badge)
                    st.progress(float(similarity) / 100)
                    st.write("") # spacing
                    
                    st.markdown("**Job Field**")
                    st.write(row["Job Field"])

                    st.markdown("**Job Description**")
                    st.write(row["Job Description"])

                    st.markdown("**Required Skills & Qualifications**")
                    st.write(row["Required Skills & Qualifications"])

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed with ❤️ using <b>Python</b> | <b>Streamlit</b> | <b>Word2Vec</b> | <b>Scikit-learn</b> | <b>Natural Language Processing</b>"
    "</div>",
    unsafe_allow_html=True
)