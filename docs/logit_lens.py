import streamlit as st
from obvspython.patchscopes import SourceContext, TargetContext, Patchscope

# Streamlit intro
st.title("Logit Lens")

# Opening paragraph
st.write(
    "This is a simple example of a logit lens using the patchscopes implementation."
    "The regular way of running the logit lens is to look at the last token of the source prompt, "
    "and see how its next-token prediction changes over the layers of the model."
    "Hence, we use a prompt where it is juuuust about to say something obvious, so when "
    "can see the moment the model groks what is expected of it."
)

# Prompt for the source prompt
source_prompt = st.text_input("Enter the source prompt", "The quick brown fox jumped over the lazy")

prediction = st.text_input("What comes next?", "dog.")
# Make sure the prediction starts with a space, and strip and punctuation
prediction = " " + prediction.strip().strip(".,!?")

# Opening paragraph
st.write(
    "Okay, now we will run the patchscope. We will show you the full output on every layer, "
    "and then we'll plot the probability of your prediction over time, to see whether the model "
    "matches it."
)

# Run the patchscope
source_context = SourceContext(prompt=source_prompt, device="cpu")
target_context = TargetContext.from_source(source_context)
target_context.layer = -1
patchscope = Patchscope(source=source_context, target=target_context)

prediction_token = patchscope.tokenizer.encode(prediction)
prediction_probs = []
for i in range(patchscope.n_layers):
    st.write(f"Running Layer {i}...")

    patchscope.source.layer = i
    patchscope.run()

    output = "".join(patchscope.tokenizer.decode(patchscope._output_tokens()))
    st.write(f"Output: {output}")

    probs = patchscope.probabilities()[-1]
    prediction_probs.append(probs[prediction_token].item())


st.write("Now we will plot the pediction probability over layers.")
st.line_chart(prediction_probs)
