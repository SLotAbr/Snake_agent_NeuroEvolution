import cma
import numpy as np
from agents import Agent
from os import mkdir
from pickle import dump, load
from tqdm import tqdm
from simulation_tool import run_simulation
from shutil import rmtree
from sys import argv


def objective_function(score_list, discount):
	return sum([score_list[t]*(discount**t) for t in range(len(score_list))])


def make_checkpoint(last_iter, opt_params, opt_metrics, population_params):
	agent_id = opt_params[1]
	path = f'history_buffer/CMA_ES/{agent_id}/optimizer_info.pkl'
	with open(path,'wb') as f:
		dump(opt_params+(last_iter,), f)

	path = f'history_buffer/CMA_ES/{agent_id}/optimizer_metrics.pkl'
	with open(path,'wb') as f:
		dump(opt_metrics, f)

	path = f'history_buffer/CMA_ES/{agent_id}/iteration_{last_iter}/iteration_info.pkl'
	with open(path,'wb') as f:
		dump(population_params, f)


def load_checkpoint(optimizer_id='CMA_ES', agent_id='Bare_minimum'):
	path = f'history_buffer/CMA_ES/{agent_id}/optimizer_info.pkl'
	with open(path, 'rb') as f:
		# es, _, DISCOUNT, ITERATION_NUMBER, iter_ = load(f)
		opt_params = load(f)

	path = f'history_buffer/CMA_ES/{agent_id}/optimizer_metrics.pkl'
	with open(path, 'rb') as f:
		# fitness_history, scores_history = load(f)
		opt_metrics = load(f)

	return opt_params, opt_metrics


AGENTS = {'Bare_minimum':103}
agent_name = 'Bare_minimum'
agent = Agent(agent_name)

if len(argv)==1:
	rmtree(f'history_buffer/CMA_ES/{agent_name}/')
	mkdir(f'history_buffer/CMA_ES/{agent_name}')
	DISCOUNT = 0.95
	es = cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5, {'popsize': 10})
	fitness_history, scores_history = [], []
	ITERATION_NUMBER, iteration = 10, 0
elif argv[1]=='load_checkpoint':
	opt_params, opt_metrics = \
		load_checkpoint(optimizer_id='CMA_ES', agent_id=agent_name)
	es, _, DISCOUNT, ITERATION_NUMBER, iteration = opt_params
	fitness_history, scores_history = opt_metrics

for iteration in range(iteration, ITERATION_NUMBER):
	population = es.ask()
	fitness_list = np.zeros(es.popsize)
	iteration_scores = []

	for i in tqdm(range(es.popsize)):
		# if iteration < 10:
		# 	population[i] +=  np.random.normal(0,4,AGENTS[agent_name])
		agent.set_genome(population[i])
		score_list, ind_scores = run_simulation((agent_name, i),('CMA_ES',iteration), agent)
		iteration_scores.append(ind_scores)
		fitness_list[i] = objective_function(score_list, DISCOUNT)
		fitness_list[i] -= 0.01*np.mean(population[i]**2) #L2 norm

	es.tell(population, fitness_list)
	fitness_history.append(max(fitness_list))
	scores_history.append(max(iteration_scores))
	print(f'max fitness score at iteration {iteration}: {fitness_history[-1]}')

	top_score_individuals = list(np.argsort(iteration_scores)[::-1])
	opt_params = (es, agent_name, DISCOUNT, ITERATION_NUMBER)
	opt_metrics = (fitness_history, scores_history)
	population_params = (population, fitness_list, top_score_individuals)
	make_checkpoint(iteration, opt_params, opt_metrics, population_params)
