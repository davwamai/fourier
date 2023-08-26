# Fourier Analysis Visualization

## Introduction

This project aims to provide a comprehensive understanding of Fourier Analysis through visualizations. It was written for the CSCE567 Visualization Tools Course at the University of South Carolina. 
The project is divided into two primary components:

1. **YouTube Video**: Created using [Manim](https://github.com/ManimCommunity/manim), a powerful tool for creating animations for explaining mathematics. [Watch it here](https://www.youtube.com/watch?v=9x4VQOlJegM).

2. **Interactive Shiny App**: Built with R, this app allows you to interact with the Fourier Analysis ecosystem, offering a more hands-on experience. [Try it out here](https://davwamai.shinyapps.io/fourier-analysis/).

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Manim Code](#running-the-manim-code)
- [Interacting with Shiny App](#interacting-with-shiny-app)

---

## Getting Started

### Prerequisites

- Python 3.x
- Manim
- R and R Shiny (for Shiny App)
  
### Installation

1. **Manim Installation**

    Install Manim using pip:

    ```bash
    pip install manim
    ```
    
2. **R and Shiny Installation**

    - Download and install R from [here](https://cran.r-project.org/mirrors.html).
    - Open R console and install the Shiny package:

    ```R
    install.packages("shiny")
    ```

---

## Running the Manim Code

1. **Clone the repository**

    ```bash
    git clone https://github.com/davwamai/fourier.git
    ```

2. **Navigate to the directory**

    ```bash
    cd fourier/manim-code
    ```

3. **Render the Scene**

    To render an arbitrary scene called `CreateCircle` from the `scene.py` file, run:

    ```bash
    manim -pql scene.py CreateCircle
    ```

    This will produce a low-quality animation for quick rendering. You can adjust the quality with different flags (`-pql`, `-pqm`, `-pqh` for low, medium, and high quality, respectively).

---

## Interacting with the Shiny App

1. **Navigate to the Shiny App directory**

    ```bash
    cd ../app
    ```

2. **Run the Shiny App**

    Open R and run:

    ```R
    shiny::runApp()
    ```

    Alternatively, you can interact with the hosted version of the Shiny App [here](https://davwamai.shinyapps.io/fourier-analysis/).

---

Feel free to star the GitHub repository if you find this project helpful! ⭐
