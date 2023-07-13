import cma
import numpy as np
from agents import Agent
from tqdm import tqdm
from simulation_tool import run_simulation


def objective_function(score_list, discount):
	return sum([score_list[t]*(discount**t) for t in range(len(score_list))])


AGENTS = {'Bare_minimum':99}
agent_name = 'Bare_minimum'
agent = Agent()

es = cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5, {'popsize': 4})
iteration_number = 10

for iteration in range(iteration_number):
	population = es.ask()
	fitness_list = np.zeros(es.popsize)

	for i in tqdm(range(es.popsize)):
		# if iteration < 10:
		# 	population[i] +=  np.random.normal(0,4,AGENTS[agent_name])
		agent.set_genome(population[i])
		score_list = run_simulation((agent_name, i),('CMA_ES',iteration), agent)
		fitness_list[i] = objective_function(score_list, 0.95)
		fitness_list[i] -= 0.01*np.mean(population[i]**2) #L2 norm

	es.tell(population, fitness_list)	
	print(f'max fitness score at iteration {iteration}: {max(fitness_list)}')
