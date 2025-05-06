# LLM Interview 

There are 2 versions of this program:
- A Streamlit version, which is quite aesthetic. It does not have sound management.
- A Tkinter version, which is ugly. It has full sound management and detection of when user starts and stops talking

## Install

Python 3.11 is required (some older versions may also work)
`pip install -r requirements`

You also need to copy-paste the `.env.dist` file to `.env`, and add your personal API key.

## Launch

To launch the Tkinter version: launch script `interface.py` in folder `tkinter`. To exit the program, control-c in the console.

To launch the Streamlit version: `streamlit run app.py` in directory `streamlit`. To stop the server, control-c in the console.

## Dummy files

Dummy job offer, resume and motivation letter can be found in directory `dummies`, in both PDF and raw text format.