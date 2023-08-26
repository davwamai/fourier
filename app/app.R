library(shiny)
library(dplyr)
library(ggplot2)
library(plotly)
library(stringr)

ui <- fluidPage(
  titlePanel("Fourier Visualization"),
  
  sidebarLayout(
    sidebarPanel(
      
      conditionalPanel(
        condition = "input.tabselected == 'Fourier Decomposition' || input.tabselected == 'Fourier Transform'",
        
        selectizeInput("freq",
                       "Frequencies:",
                       choices = 1:20,
                       selected = c(3, 7, 15),
                       multiple = TRUE),
        
        wellPanel(   
          h3("Instructions:"),
          p("Select up to ten frequencies you want to include in the signal using the selector above. Regarding the first tab, the sum of each wave
            is displayed in the top time-domain plot, and the individual sum components are displayed in the bottom plot"),
          p("You can hover over each waveform for more characteristic information, like corresponding component wave, timestep,
            and amplitude."),
          p("Try different combinations of component waves to yield unique sums (If you're having trouble understanding how these
            components sum to yield the result wave, try out my personal favorite combinations, [1 and 10] or [2 and 20] :) ")
        )
      ),
      
      conditionalPanel(
        condition = "input.tabselected == 'Fourier Series'",
        sliderInput("n", "Number of terms:", min = 1, max = 100, value = 1, step = 1),
        h3("Instructions:"),
        p("Use the slider to select approximation weight. The higher the number, the more accurate the approximation becomes"),
      ),
      
      conditionalPanel(
        condition = "input.tabselected == 'NMR Data'",
        h3("Instructions:"),
        p("Spin Echo Data for Toluene"),
        p("Click and Drag on either plot to zoom"),
        
      )
    ),

    
    mainPanel(
      tabsetPanel(type = "tabs", id = "tabselected",
                  tabPanel("Fourier Decomposition", 
                           plotlyOutput("plot", height = "400px"), 
                           plotlyOutput("componentPlot", height = "400px")),
                  tabPanel("Fourier Transform",
                           plotlyOutput("componentPlot2", height = "400px"), 
                           plotlyOutput("fourierTransform", height = "400px")),
                  tabPanel("Fourier Series",
                           plotOutput("plot2", height = "400px"),
                           textOutput("equation")),
                  tabPanel("NMR Data",
                           plotlyOutput("nmrPlot", height = "400px"), 
                           plotlyOutput("nmrFourier", height = "400px"))
      )
    )
  )
)

server <- function(input, output) {
  
  getSinusoidal <- function(f,a,p,dt = 2e-4){
    t= seq(0,1,by=dt)
    y=a*sin(2*pi*f*t+p)
    return(data.frame(y=y,t=t,f=round(f,digits=2),component=paste('Fs = ', round(f,digits=2))))
  }
  
  getSinusoidalFT <- function(f,a,p,dt = 0.01){
    t= seq(0,1,by=dt)
    y=a*sin(2*pi*f*t+p)
    return(data.frame(y=y,t=t,f=round(f,digits=2),component=paste('Fs = ', round(f,digits=2))))
  }
  
  sig.data <- reactive({
    freqs <- as.numeric(input$freq)
    amps <- rep(runif(1, min=0.1, max =1), length(freqs))
    phase <- rep(runif(1, min=0, max =2*pi), length(freqs))
    signal <- do.call('rbind', lapply(seq_along(freqs), function(x){getSinusoidal(freqs[x],amps[x],phase[x])}))
    return(signal)
  })
  
  sig.dataFT <- reactive({
    freqs <- as.numeric(input$freq)
    amps <- rep(runif(1, min=0.1, max =1), length(freqs))
    phase <- rep(runif(1, min=0, max =2*pi), length(freqs))
    signal <- do.call('rbind', lapply(seq_along(freqs), function(x){getSinusoidalFT(freqs[x],amps[x],phase[x])}))
    return(signal)
  })
  
  output$plot <- renderPlotly({
    sig.df <- tibble::as_tibble(sig.data())
    full.sig <- sig.df %>% group_by(t) %>% summarise(y = sum(y),f=factor('Full Signal'))
    p <- ggplot(full.sig, aes(x=t, y=y)) +
      geom_line() +
      theme_minimal()
    ggplotly(p)
  })
  
  output$componentPlot <- renderPlotly({
    sig.df <- tibble::as_tibble(sig.data())
    p <- ggplot(sig.df, aes(x=t, y=y, color=component)) +
      geom_line() +
      theme_minimal() +
      labs(color = 'Component Waves')
    ggplotly(p)
  })
  
  output$componentPlot2 <- renderPlotly({
    sig.df <- tibble::as_tibble(sig.data())
    p <- ggplot(sig.df, aes(x=t, y=y, color=component)) +
      geom_line() +
      theme_minimal() +
      labs(color = 'Component Waves')
    ggplotly(p)
  })
  
  output$fourierTransform <- renderPlotly({
    sig.df <- tibble::as_tibble(sig.dataFT())
    full.sig <- sig.df %>% group_by(t) %>% summarise(y = sum(y),f=factor('Full Signal'))
    ft <- abs(fft(full.sig$y)) 
    freqs <- (0:(length(ft)/2))/length(ft) * (1/0.01) 
    ft.df <- data.frame(freq = freqs, magnitude = ft[1:(length(ft)/2+1)]) 
    p <- ggplot(ft.df, aes(x=freq, y=magnitude)) +
      geom_line() +
      theme_minimal()
    ggplotly(p)
  })
  
  ###############################################################################################################################
  
  calcTerm <- function(n, t) {
    return ((4/pi) * sin((2*n-1)*t) / (2*n-1))
  }
  
  calcSeries <- function(N, t) {
    terms <- sapply(1:N, calcTerm, t = t)
    return (rowSums(terms))
  }
  
  seriesData <- reactive({
    t <- seq(-pi, pi, length.out = 200)
    y <- calcSeries(input$n, t)
    return (data.frame(t = t, y = y))
  })
  
  output$plot2 <- renderPlot({
    t <- seq(-pi, pi, length.out = 200)
    y <- sign(sin(t))
    squareWave <- data.frame(t = t, y = y)
    
    fourierData <- seriesData()
    
    equation <- str_c("Sum = ", str_c(sapply(1:input$n, function(n) paste0((4/pi)/(2*n-1), "*sin((2*", n, "-1)*t)")), collapse = " + "))
    output$equation <- renderText({equation})
    
    plot(t, y, type = "l", ylim = c(-1.5, 1.5), xlab = "Time", ylab = "Amplitude", main = "Square wave and Fourier series")
    lines(fourierData, col = "blue")
  })
###############################################################################################################################
  data <- reactive({
    df <- read.csv("spinechoes.csv", stringsAsFactors = FALSE)
    df$time <- as.numeric(as.character(df$time))
    df$mV <- as.numeric(as.character(df$mV))
    
#    print(any(is.na(df$time)))
#    print(any(is.na(df$mV)))
    
    df
  })
  
  output$nmrPlot <- renderPlotly({
    df <- data()
 #   print(head(df))  
    
 #   print(range(df$time, na.rm = TRUE))  
 #   print(range(df$mV, na.rm = TRUE))  
    
    p <- df %>%
      plot_ly(x = ~time, y = ~mV, type = 'scatter', mode = 'lines') %>%
      layout(title = "Toluene Spin Echo Data",
             xaxis = list(title = "Time"),
             yaxis = list(title = "mV"))
    p
  })
  
  output$nmrFourier <- renderPlotly({
    df <- data()
    fft_vals <- Mod(fft(df$mV))
    
    dt <- df$time[2] - df$time[1]  # time step in seconds
    fs <- 1 / dt  # sample rate in Hz
    
    N <- length(fft_vals)
    f <- fs * (0:(N/2)) / N  # frequency values in Hz
    
    fft_vals <- fft_vals[1:length(f)]  # cut off second half of FFT values
    
    p <- data.frame(frequency = f, fft_vals = fft_vals) %>%
      plot_ly(x = ~frequency, y = ~fft_vals, type = 'bar') %>%
      layout(title = "Fourier Transform",
             xaxis = list(title = "Frequency (Hz)", range = c(0, max(f))),
             yaxis = list(title = "FFT Magnitude"))
    p
  })
}

shinyApp(ui = ui, server = server)