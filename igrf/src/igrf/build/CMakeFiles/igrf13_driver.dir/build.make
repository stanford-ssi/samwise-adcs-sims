# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build

# Include any dependencies generated for this target.
include CMakeFiles/igrf13_driver.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/igrf13_driver.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/igrf13_driver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/igrf13_driver.dir/flags.make

CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o: CMakeFiles/igrf13_driver.dir/flags.make
CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o: /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13_driver.f90
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13_driver.f90 -o CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o

CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.i"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13_driver.f90 > CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.i

CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.s"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13_driver.f90 -o CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.s

CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o: CMakeFiles/igrf13_driver.dir/flags.make
CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o: /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13.f
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building Fortran object CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13.f -o CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o

CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.i"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13.f > CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.i

CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.s"
	/opt/homebrew/Caskroom/miniconda/base/bin/arm64-apple-darwin20.0.0-gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/fortran/igrf13.f -o CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.s

# Object files for target igrf13_driver
igrf13_driver_OBJECTS = \
"CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o" \
"CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o"

# External object files for target igrf13_driver
igrf13_driver_EXTERNAL_OBJECTS =

/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver: CMakeFiles/igrf13_driver.dir/fortran/igrf13_driver.f90.o
/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver: CMakeFiles/igrf13_driver.dir/fortran/igrf13.f.o
/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver: CMakeFiles/igrf13_driver.dir/build.make
/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver: CMakeFiles/igrf13_driver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking Fortran executable /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/igrf13_driver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/igrf13_driver.dir/build: /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/igrf13_driver
.PHONY : CMakeFiles/igrf13_driver.dir/build

CMakeFiles/igrf13_driver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/igrf13_driver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/igrf13_driver.dir/clean

CMakeFiles/igrf13_driver.dir/depend:
	cd /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build /Users/lundeencahilly/Desktop/github/samwise-adcs-sims/igrf/src/igrf/build/CMakeFiles/igrf13_driver.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/igrf13_driver.dir/depend

