# Project Details

This project implements one agent that interact with a Unity simulated
environment where it needs to collect the yellow bananas and avoid the
purple ones.

The agent needs to have an average score of 13 or more over 100 episodes
to consider the environment solved.

# Getting Started

The Deep_Navigator to run needs some dependencies to be installed
before.

Because it uses the Unity Engine to run the environment simulator is
necessary to install it an the ML-Agents toolkit.

Clone the project's repository local on your machine

> git clone <https://github.com/BolivarTech/Deep_Navigator.git>
>
> cd Deep_Navigator

Run the virtual environment on the console

## Windows:

> .\\venv\\Scripts\\activate.bat

## Linux:

> source ./venv/Scripts/activate

Install the following modules on python.

-   Numpy (v1.19.5+)

-   torch (v1.8.0)

-   tensorflow (v1.7.1)

-   mlagents (v0.27.0)

To install it, on the projects root you need to run on one console

> pip install .

# Instructions

To run the agent you first need to activate the virtual environment.

## Windows:

> .\\venv\\Scripts\\activate.bat

## Linux:

> source ./venv/Scripts/activate

On the repository's root you can run the agent to get a commands'
description

> python .\\main.py -h
>
> usage: main.py \[-h\] \[-t\] \[-p\] \[-m MODEL\]
>
> optional arguments:
>
> -h, \--help show this help message and exit
>
> -t, \--train Perform a model training, if -m not specified a new model
> is trained.
>
> -p, \--play Perform the model playing.
>
> -m MODEL, \--model MODEL Path to an pytorch model file.

The -t option is used to train a new model or if -m option is selected
continues training the selected model.

The -p option is used to play the agent on the environment using the
model specified on the -m flag.

The -h option shows the command's flags help

To run the model used on this try:

> python -p -m ./models/m3-18.9.pth

