# .PHONY defines parts of the makefile that are not dependant on any specific file
.PHONY = help setup test run clean environment

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = setup
# The @ makes sure that the command itself isnt echoed in the terminal
setup:
	@mkdir /mnt/disk2/samarth/temp/get-solar-eigs/efs_Jesper/snrnmais_files/eig_files
	@mkdir /mnt/disk2/samarth/temp/get-solar-eigs/efs_Jesper/snrnmais_files/data_files
	@gfortran /mnt/disk2/samarth/temp/qdPy/read_eigen.f90
	@/mnt/disk2/samarth/temp/qdPy/a.out
	@rm /mnt/disk2/samarth/temp/qdPy/a.out
	@python /mnt/disk2/samarth/temp/qdPy/mince_eig.py
	@mv *.dat /mnt/disk2/samarth/temp/get-solar-eigs/efs_Jesper/snrnmais_files/data_files/.
