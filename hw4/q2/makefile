q2:
	mpic++ q2.cpp -o ./q2.exe -std=c++11
	sbatch sbatch.script
	# sbatch 4_100_sbatch.script
	# sbatch 2_100_sbatch.script
	# sbatch 4_20_sbatch.script
	# sbatch 2_20_sbatch.script
	squeue -u $(USER)

clean:
	rm *.x *.exe *.out