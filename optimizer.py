import cma
import numpy as np
from agents import Agent
from VAE.VAE_creation import Encoder # visual cortex architecture
from os import mkdir
from pickle import dump, load
from tqdm import tqdm
from simulation_tool import run_simulation
from shutil import rmtree
from sys import argv


def objective_function(REWARD_TABLE, score_list, exploration_score, decay):
	reward = sum(
		[score_list[t]*REWARD_TABLE['scores_importance']*(REWARD_TABLE['discount']**t) \
			for t in range(len(score_list))]
	)
	reward += exploration_score*REWARD_TABLE['exploration_importance']
	reward -= decay*REWARD_TABLE['decay_intensity']
	return reward


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
		# es, _, REWARD_TABLE, ITERATION_NUMBER, iter_ = load(f)
		opt_params = load(f)

	path = f'history_buffer/CMA_ES/{agent_id}/optimizer_metrics.pkl'
	with open(path, 'rb') as f:
		# fitness_history, scores_history = load(f)
		opt_metrics = load(f)

	return opt_params, opt_metrics


AGENTS = {'Bare_minimum':103}
agent_name = 'Bare_minimum'
agent = Agent(VISUAL_CORTEX_PATH='VAE/VAE_model.pt', agent_name='Bare_minimum')


if len(argv)==1:
	rmtree(f'history_buffer/CMA_ES/{agent_name}/')
	mkdir(f'history_buffer/CMA_ES/{agent_name}')

	REWARD_TABLE = {'scores_importance':5,
					'exploration_importance':1,
					'decay_intensity':0.01,
					'discount':0.95}
	# cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5, {'popsize': 10})
	es = cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5)
	fitness_history, scores_history = [], []
	ITERATION_NUMBER, iteration = 200, 0
elif argv[1]=='load_checkpoint':
	opt_params, opt_metrics = \
		load_checkpoint(optimizer_id='CMA_ES', agent_id=agent_name)
	es, _, REWARD_TABLE, ITERATION_NUMBER, iteration = opt_params
	fitness_history, scores_history = opt_metrics

for iteration in range(iteration, ITERATION_NUMBER):
	population = es.ask()
	fitness_list = np.zeros(es.popsize)
	iteration_scores = []

	for i in tqdm(range(es.popsize)):
		# if iteration < 10:
		# 	population[i] +=  np.random.normal(0,4,AGENTS[agent_name])
		agent.set_genome(population[i])
		score_list, ind_scores, exploration_score = \
			run_simulation((agent_name, i),('CMA_ES',iteration), agent)
		iteration_scores.append(ind_scores)

		decay = np.mean(population[i]**2) #L2 norm
		reward = objective_function(
			REWARD_TABLE, score_list, exploration_score, decay
		)
		# CMAEvolutionStrategy optimizes a loss function - not a reward function
		fitness_list[i] = -reward

	es.tell(population, fitness_list)
	fitness_history.append(np.median(fitness_list))
	scores_history.append(max(iteration_scores))
	print(f'median loss value at iteration {iteration}: {fitness_history[-1]}')

	top_score_individuals = list(np.argsort(iteration_scores)[::-1])
	opt_params = (es, agent_name, REWARD_TABLE, ITERATION_NUMBER)
	opt_metrics = (fitness_history, scores_history)
	population_params = (population, fitness_list, top_score_individuals)
	make_checkpoint(iteration, opt_params, opt_metrics, population_params)
