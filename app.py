import streamlit as st

from auth import check_login
from rag import answer, current_model_label, list_local_models, load_settings, save_settings

PROVIDER_CHOICES = ["Groq", "Local (Ollama)"]
TOPICS = ["All", "earn", "acq", "crude", "trade", "money-fx", "interest", "ship", "grain"]
EXAMPLES = [
    "What is happening with OPEC and oil prices?",
    "Which companies are being acquired?",
    "What is the news about interest rates?",
    "What are the trade tensions between countries?",
]

st.set_page_config(page_title="Market Intelligence Assistant")

# Small custom style, shared by every page. Colors come from .streamlit/config.toml.
st.markdown(
    """
    <style>
    /* hide the default chat avatars; we show a small "You" / "AI" label instead */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    /* chat messages and the login form look like white cards */
    [data-testid="stChatMessage"], [data-testid="stForm"] {
        background: #FFFFFF; border: 1px solid #E2E0D8; border-radius: 8px;
    }
    [data-testid="stForm"] { padding: 1.5rem; }
    /* smaller, calmer title */
    h1 { font-size: 1.6rem; font-weight: 600; }
    /* small grey headings that split the sidebar into sections */
    .section { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #6B7280; margin: 1.2rem 0 0.3rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Makes sure every value we keep between page re-runs exists.
# WHY: Streamlit re-runs this file on every click, so session_state is our memory.
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "role" not in st.session_state:
        st.session_state.role = None


# Shows the login card and, if the password is right, remembers who signed in.
# WHY: nobody can reach the chat before signing in, and the role we save here
# decides which articles they are allowed to search.
def show_login():
    left, middle, right = st.columns([1, 2, 1])
    with middle:
        with st.form("login_form"):
            st.title("Market Intelligence Assistant")
            st.caption("Sign in to continue")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            role = check_login(username, password)
            if role is None:
                st.error("Incorrect username or password. Please check and try again.")
            else:
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()


# Turns an error into a calm, specific message for the user.
# WHY: a raw error text means nothing to a banker; we look at the kind of
# error (its class name) and say what happened and what to do next.
def friendly_error(error):
    name = type(error).__name__
    if isinstance(error, ValueError):
        return str(error)
    if "Authentication" in name or "PermissionDenied" in name:
        return "The model provider rejected the API key. Please ask the admin to check it."
    if "RateLimit" in name:
        return "The model provider is busy right now. Wait a moment and try again."
    if "Timeout" in name or "Connection" in name:
        return (
            "The model provider didn't respond. Try again, "
            "or ask the admin to switch models."
        )
    if "HTTPError" in name or "Status" in name:
        return (
            "The model provider returned an error. "
            "Please ask the admin to check the model settings."
        )
    return "The answer could not be completed. Try again, or contact the admin if it keeps happening."


# Converts a distance into a simple word the user can understand.
# WHY: a small distance means a close match, but "0.73" means little to a reader.
def relevance_label(distance):
    if distance < 0.8:
        return "High"
    if distance < 1.0:
        return "Medium"
    return "Low"


# Shows which provider/model wrote the answer, as small grey text.
# WHY: with two possible models the user should know which one replied.
def show_model(model):
    if model != "":
        st.caption("Answered by " + model)


# Shows the sources in an expander, one compact row per article.
# WHY: the user can check which articles the answer came from.
def show_sources(sources):
    with st.expander("Sources (" + str(len(sources)) + ")"):
        for s in sources:
            st.markdown(
                "**" + s["article_id"] + "** &nbsp; :blue-badge[" + s["category"] + "]"
                + " &nbsp; Relevance: " + relevance_label(s["distance"])
            )
            st.caption(s["text"][:200])


# Shows one assistant answer: the text, then its sources and model.
# WHY: used for new answers and for the saved history, so both look the same.
# No sources means the question was out of scope, so we show a neutral info box.
def show_answer(reply, sources, model):
    if len(sources) == 0:
        st.info(reply)
        return
    st.write(reply)
    show_sources(sources)
    show_model(model)


# Draws a small grey section heading in the sidebar.
# WHY: clear sections make the sidebar easy to scan.
def section(title):
    st.markdown('<div class="section">' + title + "</div>", unsafe_allow_html=True)


# Draws the model part of the sidebar.
# WHY: only the admin may switch between Groq and a local Ollama model;
# members just see which model is in use.
def draw_model_settings():
    if st.session_state.role != "admin":
        st.write(current_model_label())
        return

    settings = load_settings()
    if settings["provider"] == "ollama":
        current_index = 1
    else:
        current_index = 0
    choice = st.radio("Provider", PROVIDER_CHOICES, index=current_index)

    local_model = ""
    if choice == "Local (Ollama)":
        local_models = list_local_models()
        if len(local_models) == 0:
            st.warning(
                "Ollama is not running. Install it from ollama.com, "
                "then run: ollama pull llama3.2:3b"
            )
        else:
            model_index = 0
            if settings["ollama_model"] in local_models:
                model_index = local_models.index(settings["ollama_model"])
            local_model = st.selectbox("Local model", local_models, index=model_index)

    # Saving a local provider only makes sense when a model is selected.
    can_save = choice == "Groq" or local_model != ""
    if st.button("Save model settings", disabled=not can_save):
        if choice == "Groq":
            settings["provider"] = "groq"
        else:
            settings["provider"] = "ollama"
            settings["ollama_model"] = local_model
        save_settings(settings)
        st.success("Saved. Now using " + current_model_label())


# Draws the whole sidebar and returns the chosen topic (None means "All").
# WHY: keeps every control in one place, split into clear sections.
def draw_sidebar():
    with st.sidebar:
        section("Account")
        st.write(st.session_state.username)
        st.markdown(":blue-badge[" + st.session_state.role + "]")
        if st.button("Sign out"):
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.messages = []
            st.rerun()

        section("Search")
        topic = st.selectbox("Topic", TOPICS)
        if st.button("Clear conversation"):
            st.session_state.messages = []

        section("Model")
        draw_model_settings()

        section("About")
        st.caption(
            "Data: Reuters R8 (1987), numbers removed. "
            "Model: Groq or local Ollama (set by the admin). "
            "Embeddings: MiniLM. Vector DB: ChromaDB."
        )

    if topic == "All":
        return None
    return topic


# Shows the example questions as buttons; a click saves the question to ask.
# WHY: a new user sees what to try; the buttons disappear once the chat starts.
def show_examples():
    st.write("Try one of these:")
    columns = st.columns(2)
    for i in range(len(EXAMPLES)):
        if columns[i % 2].button(EXAMPLES[i], key="example" + str(i)):
            st.session_state.pending = EXAMPLES[i]  # asked automatically below


init_state()

# Not signed in yet: show only the login card and stop here.
if st.session_state.role is None:
    show_login()
    st.stop()

st.title("Market Intelligence Assistant")
st.caption("Answers from Reuters financial news · every answer cites its sources")

category = draw_sidebar()

# The examples live in a placeholder so we can remove them once a question is asked.
examples_area = st.empty()
if len(st.session_state.messages) == 0:
    with examples_area.container():
        show_examples()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.caption("You")
            st.write(message["content"])
        else:
            st.caption("AI")
            show_answer(message["content"], message["sources"], message["model"])

question = st.chat_input("Ask about financial news...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    examples_area.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.caption("You")
        st.write(question)

    with st.chat_message("assistant"):
        st.caption("AI")
        try:
            with st.spinner("Searching news articles..."):
                reply, sources, model = answer(
                    question, category=category, role=st.session_state.role
                )
            show_answer(reply, sources, model)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "sources": sources, "model": model}
            )
        except Exception as error:
            st.error(friendly_error(error))
