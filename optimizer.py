import cma
import numpy as np
import os
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
		# loss_history, scores_history = load(f)
		opt_metrics = load(f)

	return opt_params, opt_metrics


def write_the_best_iters_info(iter_number, 
							  iter_info,
							  min_loss_values, 
							  opt_it='CMA_ES', 
							  agent_id='Bare_minimum'):
	order = np.argsort(min_loss_values)
	sorted_iters = ' '.join([str(e) for e in np.array(iter_info)[order]])
	text = f'iteration_{iter_number} : {sorted_iters}\n'
	path = f'history_buffer/{opt_it}/{agent_id}/the_best_iters.txt'
	mode = 'a' if os.path.exists(path) else 'w'
	with open(path, mode) as f:
		f.write(text)


AGENTS = {'Bare_minimum':103,
		  '16_neurons'  :188,
		  '32_neurons'  :401,
		  '64_neurons'  :1112,
		  '128_neurons' :2527}
agent_name = 'Bare_minimum'
agent = Agent(VISUAL_CORTEX_PATH='VAE/VAE_model.pt', agent_name=agent_name)


if len(argv)==1:
	rmtree(f'history_buffer/CMA_ES/{agent_name}/')
	mkdir(f'history_buffer/CMA_ES/{agent_name}')

	REWARD_TABLE = {'scores_importance':20,
					'exploration_importance':2,
					'decay_intensity':0.01,
					'discount':0.95}
	# cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5, {'popsize': 10})
	es = cma.CMAEvolutionStrategy(AGENTS[agent_name] * [0], 0.5)
	loss_history, scores_history = [], []
	EARLY_STOPPING_THRESHOLD, is_locked = 800, False
	min_loss_values = [99999 for _ in range(10)]
	min_loss_iters =  [-1 for _ in range(10)]
	ITERATION_NUMBER, iteration = 200, 0
elif argv[1]=='load_checkpoint':
	opt_params, opt_metrics = \
		load_checkpoint(optimizer_id='CMA_ES', agent_id=agent_name)
	es, _, REWARD_TABLE, ITERATION_NUMBER, EARLY_STOPPING_THRESHOLD, \
		is_locked, min_loss_info, iteration = opt_params
	min_loss_iters, min_loss_values = min_loss_info
	loss_history, scores_history = opt_metrics

for iteration in range(iteration, ITERATION_NUMBER):
	if is_locked:
		print('the optimization interrupted due to early stopping condition')
		print('DEBUG info; (min_loss_iteration, locked_iteration, THRESHOLD):')
		iter_ = min_loss_iters[min_loss_values.index(min(min_loss_values))]
		print(f'{iter_}, {iteration}, {EARLY_STOPPING_THRESHOLD}')
		break
	population = es.ask()
	loss_list = np.zeros(es.popsize)
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
		loss_list[i] = -reward

	es.tell(population, loss_list)
	loss_history.append(np.median(loss_list))
	scores_history.append(max(iteration_scores))
	print(f'median loss value at iteration {iteration}: {loss_history[-1]}')

	sort_indexes = np.argsort(loss_list)
	for i in sort_indexes:
		candidate = max(min_loss_values)
		if loss_list[i] < candidate:
			x = min_loss_values.index(candidate)
			min_loss_iters[x] = iteration
			min_loss_values[x] = loss_list[i]
		else:
			break
	
	if iteration%1000==0:
		write_the_best_iters_info(
			iteration, min_loss_iters, min_loss_values, 'CMA_ES', agent_name
		)

	if iteration - min_loss_iters[min_loss_values.index(min(min_loss_values))]\
		  > EARLY_STOPPING_THRESHOLD:
		is_locked = True

	top_score_individuals = list(np.argsort(iteration_scores)[::-1])
	min_loss_info = (min_loss_iters, min_loss_values)
	opt_params = (es, agent_name, REWARD_TABLE, ITERATION_NUMBER, 
		EARLY_STOPPING_THRESHOLD, is_locked, min_loss_info)
	opt_metrics = (loss_history, scores_history)
	population_params = (population, loss_list, top_score_individuals)
	make_checkpoint(iteration, opt_params, opt_metrics, population_params)
