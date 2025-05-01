import time
import streamlit as st
import pandas as pd
import random
import re
import json
from streamlit_sortables import sort_items

from pyairtable import Api

airtable_config_path = "src/user_study/airtable_config.json"
airtable_config = json.load(open(airtable_config_path))
API_KEY = open(airtable_config["key"], "r").read().strip()

api = Api(API_KEY)
annotations_table = api.table(airtable_config['annotation_table']['base_id'],
                              airtable_config['annotation_table']['table_name'])
batches_table = api.table(airtable_config['batches_table']['base_id'], airtable_config['batches_table']['table_name'])
baselines = ['random', 'ours', 'gpt-4o', 'sciIE', 'mpnet_zero', 'positive']


def build_query(anchor_text, relation):
    if relation == 'combination':
        query = f"What could we blend with **{anchor_text}** to address the context?"
    else:
        anchor_text = anchor_text.capitalize()
        query = f"In this context, what would be a good source of inspiration for **{anchor_text}**?"

    return query


def send_to_airtable(df):
    batch_id = st.session_state.batch_id
    chunk_size = 10
    for idx in range(0, len(df), chunk_size):
        chunk = df.iloc[idx:idx + chunk_size]
        records = []
        for _, row in chunk.iterrows():
            record = {
                "example_id": row['id'],
                "batch_id": batch_id,
                "context": row['context'],
                "query": row['query'],
                "gold": row['gold'],
                "annotator": row['annotator'],
                "is_ill_defined": row['is_ill_defined'],
                "knowledge_level": str(row['knowledge_level']),
            }
            baselines_results = row['baselines_results']
            record['baselines_results'] = json.dumps(baselines_results)
            records.append(record)
        annotations_table.batch_create(records)

    records = batches_table.all(formula=f"{{batch_id}} = '{batch_id}'")
    if records:
        record_id = records[0]['id']
        batches_table.update(record_id, {"status": "done"})
        print(f"Batch {batch_id} marked as Completed.")
    else:
        print(f"Batch {batch_id} not found.")


def get_user_data_chunk(user_email):
    records = batches_table.all()
    records = [record for record in records if record['fields'].get("status") in ["not_started", "in_progress"]]
    records = [record for record in records if record['fields'].get("annotator") == user_email]
    records = sorted(records, key=lambda x: int(x['fields']['priority']))
    for record in records:
        record_id = record['id']
        batch_id = record['fields'].get("batch_id")
        batches_table.update(record_id, {"status": "in_progress"})
        batch = record['fields'].get("data")
        batch = pd.DataFrame(json.loads(batch))
        st.session_state.batch_id = batch_id
        print(f"Assigned batch {batch_id} to {user_email}.")
        return batch

    print("No available batches to assign.")
    return None


# Initialize session state variables
if "email_entered" not in st.session_state:
    st.session_state.email_entered = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "data_chunk" not in st.session_state:
    st.session_state.data_chunk = None
if "batch_id" not in st.session_state:
    st.session_state.batch_id = ""
if 'save_path' not in st.session_state:
    st.session_state.save_path = ""
if "current_example" not in st.session_state:
    st.session_state.current_example = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "shuffled_baselines" not in st.session_state:
    st.session_state.shuffled_baselines = {}
if "finished" not in st.session_state:
    st.session_state.finished = False

# Email entry screen
if not st.session_state.email_entered:
    # Page title
    st.markdown("<h1 style='text-align: center;'>Welcome 🤗</h1>", unsafe_allow_html=True)
    st.write("<h3 style='text-align: center;'>Thank you for contributing to our research!</h3>", unsafe_allow_html=True)
    st.markdown(
        "This platform is designed to collect **annotations on the helpfulness of AI-generated ideas** for our study.")

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("<h6 style='text-align: left;'>Enter your email to begin:</h6>", unsafe_allow_html=True)
        with st.form(key="email_form"):
            email = st.text_input("**Email Address**", key="email_input", placeholder="Enter your email...")


            def is_valid_email(email):
                return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email))  # Regex validation


            submit = st.form_submit_button("🚀 Submit")

            if submit:
                if is_valid_email(email):
                    username = email.split("@")[0]
                    st.session_state.email_entered = True
                    st.session_state.user_email = email
                    st.session_state.user_name = username

                    # Fetch user-specific data
                    st.session_state.data_chunk = get_user_data_chunk(email)
                    if st.session_state.data_chunk is None or st.session_state.data_chunk.empty:
                        st.warning(
                            "⚠️ You've finished all your tasks. Please contact the study coordinator (Noy) if that's a problem.")
                        st.session_state.email_entered = False
                        st.session_state.user_email = ""
                        st.session_state.user_name = ""
                        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                        st.markdown("Redirecting back to login...")
                        time.sleep(3)  # Give time to see the message
                        st.rerun()
                    else:
                        st.success(f"✅ Welcome, {username}! Redirecting you now...")
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter a valid email (e.g., user@example.com).")

elif not st.session_state.finished:
    current_example = st.session_state.current_example
    example = st.session_state.data_chunk.iloc[current_example]

    if current_example not in st.session_state.shuffled_baselines:
        random.shuffle(baselines)
        st.session_state.shuffled_baselines[current_example] = baselines
    else:
        baselines = st.session_state.shuffled_baselines[current_example]
    st.subheader("Task: Evaluating AI-Generated Ideas", divider="blue")
    st.info(
        "##### ℹ️ Guidelines\n\n"
        "###### Please read the guidelines carefully before you start.\n\n"
        "Your goal is to assess how helpful AI-generated suggestions are in helping researchers generate interesting ideas and gain fresh perspectives.\n\n"
        "You will be provided with:\n"
        "1. **A context** describing the problem, specific settings, goal, etc.\n"
        "2. **A query** requesting a suggestion relevant to the context.\n"
        "3. **A list of AI-generated suggestions**.\n\n"
        "**Rank the suggestions based on how helpful they are for generating interesting ideas**. Consider the following:\n"
        " - Is the suggestion thought provoking and interesting?\n"
        " - Does it address the query and fit the context?\n"
        " - Is it clear and actionable?\n\n"
    )

    # Example Section
    with st.expander("📝 **Toy Example**", expanded=False):
        st.markdown("""
        **Context**\n
        Recent advancements in video generation have struggled to model complex narratives and maintain character consistency.

        **Query**\n
        In this context, what would be a good source of inspiration for **video generation**?

        **Suggestions**\n
        * The human brain
        * Movie production techniques, which breaks a complex narrative into a series of scenes
        * Image generation models
        * Probabilistic graphs 

        **Ranking**\n
        I would rank the suggestions as follows:
        1. Movie production techniques, which breaks a complex narrative into a series of scenes _(interesting, fits the query, clear and actionable)_
        2. The human brain _(interesting, however, this is more vague and less actionable)_
        3. Image generation models _(perhaps similar works might provide some insights to the problem, but this it's a very interesting suggestion)_
        4. Probabilistic graphs _(interesting, but it's not clear how it relates to the query)_
        """)

    # FAQ Section
    with st.expander("📚 **Frequently Asked Questions**", expanded=False):
        st.markdown("""
        ###### Q: Some suggestions are really bad, what should I do with them?
        A: That's okay, we're comparing multiple baselines, some of them are relatively weak. We're interested in the relative performance of the baselines, so if you think a suggestion is particularly bad, put it in a lower position.
        
        ###### Q: I don't understand the context or the meaning of some of the suggestions.
        A: If that's something small (like a particular term you're unfamiliar with), google it. Make sure your knowledge 
                    level reflects your actual understanding of the example.
        """)

    anchor = example['anchor']
    relation = example['relation']
    context = example['context']
    query_text = build_query(anchor, relation)

    with st.form(key=f'form_{current_example}'):

        st.markdown('##### Context')
        st.markdown(f"{context[0].capitalize() + context[1:]}")
        st.markdown("##### Query")
        st.markdown(f"{query_text[0].capitalize() + query_text[1:]}")
        st.markdown('##### Suggestions')
        st.markdown("**Drag and drop the suggestions to rank them according to the guidelines.**")

        suggestions = []
        for i, baseline in enumerate(baselines, start=1):
            suggestion = example[baseline]
            suggestions.append(f'{example[baseline].capitalize()}')

        custom_css = """
        .sortable-container {
            border: 1px dashed #ccc;
            padding: 10px;
            border-radius: 10px;
        }
        
        .sortable-item {
            # padding: 10px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border: 2px solid #ddd;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin: 10px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            cursor: grab;
            color: black;  /* Ensures text is visible */
            # transition: background-color 0.3s ease;  /* Smooth transition */
        }
        """

        ranked_suggestions = sort_items(suggestions, direction='vertical', custom_style=custom_css)

        st.divider()

        st.markdown('**Is the query ill-defined?**')
        st.markdown(
            "Mark this option if the query does not make sense. Afterwards, you can submit the form as is, we'll ignore your answer.")
        is_ill_defined = st.toggle(label="**The query is ill-defined**")

        st.divider()

        st.markdown('**How knowledgeable are you in this area?**')
        st.markdown("**1** - Nothing beyond general knowledge, this isn't my area of expertise.")
        st.markdown("**5** - Extremely knowledgeable, deeply familiar with this area.")
        knowledge_level = st.slider(
            label="knowledge_level",
            label_visibility="collapsed",
            min_value=1, max_value=5, value=3, step=1,
            help="**1** - No specific knowledge beyond the general domain. \n"
                 "**5** - Extremely knowledgeable, deeply familiar with this area."
        )

        st.divider()

        columns = st.columns([4, 1, 2])
        with columns[2]:
            submitted = st.form_submit_button("Submit & Proceed ➡️")
        baselines_results = {}
        for baseline in baselines:
            rank = ranked_suggestions.index(example[baseline].capitalize()) + 1
            baselines_results[baseline] = {
                'suggestion': example[baseline],
                'k': str(example['k']),
                'rank': rank
            }

        if submitted:
            annotations = {
                "id": example["id"],
                "annotator": st.session_state.get("user_email", "anonymous"),
                'context': context,
                'query': query_text,
                'gold': example['positive'],
                'is_ill_defined': is_ill_defined,
                'knowledge_level': knowledge_level,
                "baselines_results": baselines_results
            }

            st.session_state.annotations.append(annotations)
            st.session_state.current_example += 1 if current_example < len(st.session_state.data_chunk) - 1 else 0
            st.session_state.finished = current_example >= len(st.session_state.data_chunk) - 1
            st.rerun()

        st.progress((current_example + 1) / len(st.session_state.data_chunk))
        st.markdown(f"Task {current_example + 1} of {len(st.session_state.data_chunk)}")

else:
    st.markdown("<h2 style='text-align: center;'>🎉 You have completed all your tasks! 🎉</h2>", unsafe_allow_html=True)

    # Center the thank-you message
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h3 style='text-align: center;'>Thank you for your contribution! 🥹</h3>", unsafe_allow_html=True)

    # Fetch user annotations
    annotations_df = pd.DataFrame(st.session_state.get("annotations", []))

    st.divider()
    st.balloons()  # 🎈 Confetti effect!
    send_to_airtable(annotations_df)
