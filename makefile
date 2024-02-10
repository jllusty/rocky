.DEFAULT_GOAL := main.cpp

all: clean main.cpp run

main.cpp: 
	mpic++ -I "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers" -std=c++20 -o ./bin/rocky.out ./src/rocky.cpp

run: main.cpp
	mpirun -np 8 ./bin/rocky.out

clean: 
	rm -rf ./bin/*.out