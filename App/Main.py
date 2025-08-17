import streamlit as st
import io
import sys
import os
import time
import threading
from queue import Queue
# For debugging enabling sequential execution
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import traceback
from config import hftk,Model_path,tokeniser_path,base_model
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    get_peft_model_state_dict,
#    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel
)
from accelerate import Accelerator
from utils import (
    generate_response,
    base_prompt_format,
    code,
    ins,
    code_prompt_format,
    generate_answer_parllel
)
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

# Set the page title and layout
st.set_page_config(page_title="Python IDE", layout="wide")

# Cache model and tokenizer loading
@st.cache_resource
def load_model(base_model,Model_path):
  '''
  For Loading model into GPU and the function is cached to prevent model getting reloaded
  everytime streamlit reloads
  '''
  # Explicitly set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #Initialize Accelerator
  accelerator = Accelerator(device_placement=True)

  lora_config = PeftConfig.from_pretrained(Model_path)

  quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype = torch.float16)

  model = AutoModelForCausalLM.from_pretrained(base_model,quantization_config=quantization_config,
  device_map="auto",force_download=True)

  model = get_peft_model(model, lora_config)

  model = PeftModel.from_pretrained(model, Model_path)

  #model.to(device)

  model = accelerator.prepare(model)

  return model

# Cache model and tokenizer loading
@st.cache_resource
def load_tokenizer(tokeniser_path):
  '''
  Loding tokeniser and it is cached to prevent reloading of model everytime streamlit
  is reloading
  '''
  loaded_tokenizer = AutoTokenizer.from_pretrained(tokeniser_path)
  loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
  loaded_tokenizer.pad_token_id = loaded_tokenizer.eos_token_id
  return loaded_tokenizer


#setting hugging face token
os.environ["HF_TOKEN"] = hftk
def input_page():
    '''
    Screen 1 for getting coding question from learner.
    '''
    st.markdown('<h5>Enter the problem below</h5>',unsafe_allow_html=True)
    question = st.text_area("Ask question to LLM",height=100,label_visibility='collapsed')

    # Create a button to run the code
    sub_question = st.button("Submit Question")

    if sub_question:
        # Navigate to the second page after user submits the form
        #Storing question in session state
        st.session_state['ques'] = question
        st.session_state['Gen_response'] = False
        st.session_state.page = "console"


def console_page(model,tokeniser):
    '''
    Secound screen
    '''
    ques = st.session_state.get('ques', 'No question provided')
    #Generating logical steps for coding problem
    if not st.session_state.get('Gen_response',''):
        #Generating prompt including the given question
        Resp_prompt=base_prompt_format(ques)
        #Generating Logical steps by calling model
        response=generate_response(Resp_prompt,True,tokeniser,model)
        #Moving the generated response to response session state variable
        st.session_state['response'] = ins(response)
        #Updating the generated response to response variable
        response = st.session_state['response']
        #updating session state
        st.session_state['Gen_response'] = True
    else:
        #Setting previously generated response back to response variable to prevent deletion of generated
        #response during streamlit restart
        response = st.session_state.get('response', '')

    #title of the page
    st.markdown('<h1 style="text-align: center;">Python Guide</h1>', unsafe_allow_html=True)
    res = st.button("Reset")

    if res:
        st.session_state.page = "input"  # switch to input page
        st.session_state['ques'] = ''  # Clear any previous question if necessary
        st.experimental_rerun()  # Force a rerun to navigate back

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h5>Write your Python code here</h5>',unsafe_allow_html=True)
        code = st.text_area("Python code: ",height=300,label_visibility='collapsed')
        # Create a button to run the code
        run_button = st.button("Run Code")
        # Area for program output
        st.subheader("Output")
        if run_button:
            output_queue = Queue()
            # Redirect stdout to capture print statements
            def execute_code():
                old_stdout = sys.stdout
                new_stdout = io.StringIO()
                sys.stdout = new_stdout

                try:
                    exec_globals = {}
                    # Execute the code
                    exec(code, exec_globals)
                    # Get standard output and display it
                    output = new_stdout.getvalue()
                    output_queue.put(output)

                except Exception as e:
                    # Capture and display the traceback for errors
                    traceback.print_exc(file=new_stdout)
                    # Get the error from stdout
                    error_output = new_stdout.getvalue()
                    # Push the error into output queue
                    output_queue.put(error_output)

                finally:
                    # Reset stdout
                    sys.stdout = old_stdout
            # Declare a thread to run the code
            execution_thread = threading.Thread(target=execute_code)
            # Run the thread
            execution_thread.start()
            # Display the output of code
            st.text(output_queue.get())

    with col2:

        # Create a text area for LLM response
        st.markdown('<h5>Logical Steps</h5>',unsafe_allow_html=True)

        # Display the steps generated from LLM
        out = st.text_area("Response", value=response, height=300, disabled=True,label_visibility='collapsed')

        # Declare a text area to get learners question
        question = st.text_area("Ask question to LLM (Dont ask direct code for problem)",height=100,label_visibility='collapsed')

        col1, col2 = st.columns(2)

        with col1:
            # Create a button to generate response again
            Gen_resp_ag_ques = st.button("Gen Response again")

        with col2:
            # Create a button to ask the question
            ask_ques = st.button("Ask Question")
            # Button to clear the cached model
            if st.button("Clear Cached Model"):
                st.cache_resource.clear()
                st.success("Model cache cleared!")

        if Gen_resp_ag_ques:
            st.session_state['Gen_Resp_ag_flag'] = True
        else:
            st.session_state['Gen_Resp_ag_flag'] = False

        Gen_Resp_ag_flag = st.session_state.get('Gen_Resp_ag_flag', '')

        if Gen_Resp_ag_flag:
            Resp_prompt = base_prompt_format(ques + "Consider bellow suggestion provided by the learner if it is correct" + question)
            response=generate_response(Resp_prompt,True,tokeniser,model)
            st.session_state['response'] = ins(response)


        if ask_ques:
            st.session_state['Ask_ques_flag'] = True
        else:
            st.session_state['Ask_ques_flag'] = False

        Ask_ques_flag = st.session_state.get('Ask_ques_flag', '')

        if Ask_ques_flag:
            st.session_state['Ask_ques'] = "Generating response, please wait..."

            # Create a queue to get the result from the background thread
            result_queue = Queue()

            try:
                print("Starting task...")
                answer = generate_answer_parllel(ques,str(response),question,tokeniser,model,code)
                # Push the generated response to queue
                result_queue.put(answer)
                print(result_queue)
                print("Task completed!")
            except Exception as e:
                # Push the error inside the queue
                result_queue.put(f"Error: {str(e)}")

            st.session_state['Ask_ques'] = result_queue.get()

            st.session_state['Ask_ques_flag'] = False

            ques_resp = st.session_state.get('Ask_ques', 'Generating response...')
        else:
               try:
                # Get the response from queue
                ques_resp = result_queue.get()
               except:
                ques_resp = st.session_state.get('Ask_ques','')

        # Print the response
        st.text(ques_resp)




# Main app logic
def main():
    model=load_model(base_model,Model_path)
    tokeniser=load_tokenizer(tokeniser_path)
    # Set default page
    if "page" not in st.session_state:
        st.session_state.page = "input"  # Default to input page

    # Switching between pages based on the session state
    if st.session_state.page == "input":
        input_page()
    elif st.session_state.page == "console":
        console_page(model,tokeniser)


# Run the app
if __name__ == "__main__":
    main()
