Project information

1.  The project has a dependency on OpenCV.  I have used OpenCV 3.4.1 so the project 
files will contain references to this version of the libraries.  You can use any version of
OpenCV, although it is tested with OpenCV 3.x (3.0 and later).  Because the functionality is
very limited, mainly for reading/writing files (and converting to arrays), earlier versions
should work, but it has not been tested.

2.  In the project, there is a property sheet file (OpenCV.props) where macros are defined 
to allow the user to set OpenCV related paths (include files/libraries).  You may wish to 
open this to adjust the paths to the configuration you have.

3. The project files, *.sln and *.vcxproj files, are for Visual Studio 2017.  You can download
the community edition if you do not have it.  If you have VS 2019, you can upgrade the solution
and project files.  Unfortunately, if you have an earlier version, you will have to install the 
more recent Visual Studio versions (there are community editions which are free) or you have
to create your own project files.  Please note that the solution contains the GPULib project,
which builds a dynamic link library (DLL; or *.so file if you are more familiar with the Linux
nomenclature - for shared object) and the executable project that loads the GPULib.dll at 
loading of the executable.

4.  There is, obviously, also a dependency on NVIDIA CUDA.  On my system, I have used CUDA 10.1.
I have not used any of the more advanced features of the GPU/CUDA library, so older GPUs should
be able to run this (you may just need to adjust the dependencies in the project files to point
to the directories where you have installed CUDA).

5.  I have currently done a quick check with both the debug and release versions of the builds using
a clean platform.  It built with no problems.  However, it is important to remember that when OpenCV
and CUDA were installed, environment variables were set that my projects rely on.  Different versions
of CUDA and OpenCV libraries may have different environment variables so you may need to adjust the
project files to allow for differences in library versions (OpenCV/CUDA).

6. I have put annotations in the header files.  I am still hoping to put this into a set of notes
with diagrams and add some specific CUDA/OpenCL version code, but in order to make this available
more quickly, I have not yet done so.  Actually, there may be an interest in a more detailed course
on this at my current employer, so I may expand these notes into a course rather than an overview 
on GPU and image processing.

7.  If you have questions, please feel free to get in touch.  Initially, you may contact Prof. Chinthaka as  
he can put you in touch with me.  I am quite happy to have you contact me directly, but so that I do not get 
unexpected e-mails that I (or rather, my anti-malware software) do not recognise, which may get lost, it would be 
better that I am aware to expect contact.  He is free to give you my e-mail but, as I mentioned, it would 
help me if I am aware of incoming queries.

I hope this is of help to you.  It is a very superficial introduction to the topic, but I hope it helps you
explore whether this might be a tool that is relevant to your work.
