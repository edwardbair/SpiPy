# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Source files
SOURCES = test.cpp

# Executable name
EXECUTABLE = test

# NLopt include directory
NLOPT_INCLUDE = /opt/homebrew/Cellar/nlopt/2.7.1/include

# NLopt library directory
NLOPT_LIB = /opt/homebrew/Cellar/nlopt/2.7.1/lib

# Default target
all: $(EXECUTABLE)

# Compile the program
$(EXECUTABLE): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ -I$(NLOPT_INCLUDE) $(SOURCES) -L$(NLOPT_LIB) -lnlopt

# Clean up
clean:
	rm -f $(EXECUTABLE)
