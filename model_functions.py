import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import discount, prob_time_interval

STAGE_ORDER = ['F0', 'F1', 'F2', 'F3', 'F4', 'HCC', 'DC', 'LT', 'post-LT', 'Death']
# TODO Add in 3_mo_post_LT, 6_mo_post_LT, 12_mo_post_LT??

MORTALITY_COLUMNS = ['Death']
# TODO split Death into liver-related, all-cause, cardiovascular?

DIR_HRP = "./" # TODO global variable

# TODO make more flexible? Allow passing in qaly table vs calling from file?
def load_costs(file_path, treatment_type='SOC', col_suffix='', tx_cost_override=None):
    """
    Load cost data from files and return cost structures.

    :param file_path: str, path to cost_input csv file.
    :param treatment_type: str, type of treatment to get cost for (e.g., 'SOC', 'Semaglutide').
    :param col_suffix: str, suffix to append to cost column names ['', '_low', '_high'].
    :param tx_cost_override: int or None, if provided, overrides treatment cost from file.
    :return: (pd.Series, pd.Series, float) stage_costs, background_age_costs, treatment_cost
    """
    costs = pd.read_csv(file_path, index_col='Stage')
    background_costs = pd.read_csv(os.path.join(DIR_HRP, "background_mortality.csv"), index_col='Age')

    treatment_cost = costs.loc[treatment_type, 'Cost' + col_suffix]
    if tx_cost_override is not None and treatment_type != 'SOC':
        treatment_cost = tx_cost_override

    age_costs = background_costs.loc[:, "background_cost_2025" + col_suffix]

    # TODO calculate rate and cost of LT complications

    # return stage_costs (length STAGE_ORDER),
    # age_costs (from 12-100),
    # treatment cost
    return costs.loc[STAGE_ORDER, 'Cost' + col_suffix], age_costs, treatment_cost

# TODO make more flexible? Allow passing in qaly table vs calling from file?
def load_qalys(file_path, stage_suffix='', age_suffix=''):
    """Load QALY data from a CSV file."""
    qalys = pd.read_csv(file_path, index_col='Stage')

    # stage_qalys are assumed to be presented as (negative) adjustments to age-related qalys
    stage_qalys = qalys.loc[STAGE_ORDER, 'QALY' + stage_suffix]

    # get age related indices from df
    age_indices = np.arange(12, 101).astype(str)
    select_age_qalys = qalys.loc[qalys.index.isin(age_indices), 'QALY' + age_suffix]
    age_qalys = pd.Series(index=age_indices, dtype=float)
    age_qalys.loc[select_age_qalys.index] = select_age_qalys.values
    age_qalys.ffill(inplace=True)
    age_qalys.index = age_qalys.index.astype(int)

    # repeat age_qalys series for each stage as columns
    # then subtract stage_qalys from each row as proportions
    combined_qalys = pd.DataFrame(
        np.repeat(age_qalys.values, len(STAGE_ORDER)).reshape(len(age_qalys), len(STAGE_ORDER)),
        index=age_qalys.index, columns=STAGE_ORDER)

    # stage qalys should be treated as proportion decrease from Healthy, applied to age qalys
    combined_qalys *= (1 + stage_qalys.values)

    # Override death value
    combined_qalys['Death'] = 0

    return stage_qalys, age_qalys, combined_qalys


def gen_risk_matrix(risk_path=None, regression_risk=1, progression_risk=1):
    if risk_path is not None:
        risk_df = pd.read_csv(risk_path, index_col=['from_state', 'to_state'])
        return risk_df['relative_risk'].unstack().fillna(1)
    else:
        risk_matrix = pd.DataFrame(1.0, index=STAGE_ORDER[:-1], columns=STAGE_ORDER)
        for from_state in STAGE_ORDER[:-4]:  # leaves out DC, LT, LT-post and Death  # TODO leave out F4***
            for to_state in STAGE_ORDER[:-3]:  # leaves out LT, LT-post and Death
                if STAGE_ORDER.index(to_state) < STAGE_ORDER.index(from_state):
                    risk_matrix.loc[from_state, to_state] = regression_risk
                elif STAGE_ORDER.index(to_state) > STAGE_ORDER.index(from_state):
                    risk_matrix.loc[from_state, to_state] = progression_risk
        return risk_matrix

def load_transition_matrix(transition_path, col_suffix, cycle_length=1/4):
    """Load transition matrix and make matrix.

    :param transition_path: str, path to transition matrix CSV file
    :param cycle_length: length of each cycle in years
    :return: transition matrix adjusted for cycle length
    """
    transition_matrix = pd.read_csv(transition_path)

    missing_prob_mask = transition_matrix['annual_transition_prob'].isnull()
    transition_matrix.loc[missing_prob_mask, 'annual_transition_prob'] = (1 - np.exp(-transition_matrix.loc[missing_prob_mask, 'incidence' + col_suffix] / 100)).values

    transition_matrix = transition_matrix[['from_state', 'to_state', 'annual_transition_prob']].set_index(
        ['from_state', 'to_state']).unstack().fillna(0)
    transition_matrix.columns = transition_matrix.columns.droplevel()
    transition_matrix.loc['Death', 'Death'] = 0

    # add an LT -> post-LT transition with probability 1 - LT Death probability
    transition_matrix.loc['LT', 'post-LT'] = 1 - transition_matrix.loc['LT', 'Death']
    transition_matrix.loc['post-LT', 'post-LT'] = 0

    transition_matrix = transition_matrix.loc[STAGE_ORDER, STAGE_ORDER]

    transition_matrix.fillna(0, inplace=True)

    # adjust transition matrix for cycle length
    transition_matrix_cycle = transition_matrix.map(lambda prob: prob_time_interval(prob, intervals=1/cycle_length))

    return transition_matrix_cycle


def gen_transition_probabilities(transition_matrix, age, overall_mortality, risk_matrix=None, cycle_length=1/4):
    """Generate transition probabilities for a given age and risk matrix."""
    if risk_matrix is not None:
        # multiply base transition matrix by risk matrix
        transition_matrix.update(transition_matrix * risk_matrix)

    # Convert annual overall mortality to the correct cycle probability
    annual_mortality = overall_mortality.loc[age, 'overall_mortality_avg']
    cycle_mortality = prob_time_interval(annual_mortality, intervals=1/cycle_length)

    # Add in overall mortality
    transition_matrix.iloc[:-1, -1] = transition_matrix.iloc[:-1, -1] + cycle_mortality

    # adjust LT -> post-LT transition based on new LT -> Death probability
    transition_matrix.loc['LT', 'post-LT'] = 1 - transition_matrix.loc['LT', 'Death']

    remainders = 1 - transition_matrix.sum(axis=1)
    np.fill_diagonal(transition_matrix.values, remainders)

    return transition_matrix


def run_markov_cohort(base_transition_matrix, init_state, n_cycles, cycle_length, starting_age=12,
                      risk_matrix=None, risk_attenuated_cycles=None, risk_decreases=False, begin_tx_at_cycle=0,
                      drug_stages=None):
    n_states = len(STAGE_ORDER)
    markov_trace = np.zeros((n_cycles + 1, n_states))
    markov_trace[0, :] = init_state

    overall_mortality = pd.read_csv(os.path.join(DIR_HRP, "background_mortality.csv"), index_col=0)

    if drug_stages is not None:
        for stage in STAGE_ORDER:
            if stage not in drug_stages:
                risk_matrix.loc[stage, :] = 1.0  # TODO

    for t in range(n_cycles):
        # modify risk matrix

        if (t < begin_tx_at_cycle or
                (risk_attenuated_cycles is not None and t > begin_tx_at_cycle + risk_attenuated_cycles)):
            current_risk_matrix = None
        # apply attenuation to risk_matrix
        elif risk_matrix is not None and risk_decreases:
            assert risk_attenuated_cycles is not None, "risk_attenuated_cycles must be specified if risk_decreases is True"
            # applies a linear decrease to the risk over risk_attenuated_cycles
            current_risk_matrix = (risk_matrix - 1) * (1 - min(t, risk_attenuated_cycles) / risk_attenuated_cycles) + 1
        else:
            current_risk_matrix = risk_matrix

        age = starting_age + t * cycle_length
        t_matrix = gen_transition_probabilities(base_transition_matrix.copy(), np.floor(age),
                                                overall_mortality,
                                                current_risk_matrix,
                                                cycle_length)

        markov_trace[t + 1, :] = np.dot(markov_trace[t, :], t_matrix)

    trace_df = pd.DataFrame(markov_trace, columns=STAGE_ORDER)
    trace_df['age_float'] = np.linspace(starting_age, starting_age + n_cycles * cycle_length, n_cycles + 1)
    trace_df['age'] = np.floor(trace_df['age_float'])

    return trace_df


def run_mc_sim(base_transition_matrix, init_state, n_cycles, n_iter, starting_age=12, cycle_length=1, risk_matrix=None):
    assert len(init_state) == len(STAGE_ORDER), "Initial state vector length must match number of states."
    assert len(init_state) == len(base_transition_matrix), "Transition matrix size must match number of states."

    overall_mortality = pd.read_csv(os.path.join(DIR_HRP, "background_mortality.csv"), index_col=0)

    markov_trace = np.zeros((n_iter, n_cycles + 1))
    for i in range(n_iter):
        starting_state = np.random.choice(len(init_state), p=init_state)
        markov_trace[i, 0] = starting_state
        for t in range(n_cycles):
            age = starting_age + t * cycle_length
            t_matrix = gen_transition_probabilities(base_transition_matrix.copy(), np.floor(age),
                                                    overall_mortality,
                                                    risk_matrix,
                                                    cycle_length)
            # print(t_matrix)
            markov_trace[i, t + 1] = np.random.choice(len(init_state), p=t_matrix.iloc[int(markov_trace[i, t]), :].values)

    trace_df = pd.DataFrame(markov_trace).T
    trace_df['age_float'] = np.linspace(starting_age, starting_age + n_cycles * cycle_length, n_cycles + 1)
    trace_df['age'] = np.floor(trace_df['age_float'])
    trace_df.set_index(['age_float', 'age'], inplace=True)

    return trace_df.T


# TODO merge with above
def run_mc_sim_complex(base_transition_matrix, init_state, n_cycles, n_iter, starting_age=12, cycle_length=1., risk_matrix=None, begin_tx_at_cycle=0, tx_cycle_num=None, drug_stages=None):
    assert len(init_state) == len(STAGE_ORDER), "Initial state vector length must match number of states."
    assert len(init_state) == len(base_transition_matrix), "Transition matrix size must match number of states."

    if tx_cycle_num is None:
        tx_cycle_num = n_cycles

    markov_trace = np.zeros((n_iter, n_cycles + 1))
    on_drug = np.zeros((n_iter, n_cycles + 1))

    overall_mortality = pd.read_csv(os.path.join(DIR_HRP, "background_mortality.csv"), index_col=0)

    # keeping track of this assumes drug only has given efficacy over lifetime (two separate treatments of 36 months is similar to single tx of 72 months). Could change this to only track contiguous treatments.
    num_drug_cycles = np.zeros(n_iter)
    for i in range(n_iter):
        starting_state = np.random.choice(len(init_state), p=init_state)
        markov_trace[i, 0] = starting_state
        for t in range(n_cycles):
            age = starting_age + t * cycle_length

            local_risk_matrix = risk_matrix.copy()
            this_stage_idx = int(markov_trace[i, t])
            this_stage = STAGE_ORDER[this_stage_idx]

            if this_stage in MORTALITY_COLUMNS:
                markov_trace[i, t + 1:] = this_stage_idx
                break
            # transform risk_matrix based on
            # drug stages
            local_on_drug = (((drug_stages is not None and this_stage in drug_stages) or
                              (drug_stages is None)) and
                             begin_tx_at_cycle <= t and
                             num_drug_cycles[i] < tx_cycle_num)
            if local_on_drug:
                on_drug[i, t] = 1
                num_drug_cycles[i] += 1
            else:
                local_risk_matrix = None

            t_matrix = gen_transition_probabilities(base_transition_matrix.copy(), np.floor(age),
                                                    overall_mortality,
                                                    local_risk_matrix,
                                                    cycle_length)
            # print(t_matrix)
            markov_trace[i, t + 1] = np.random.choice(len(init_state), p=t_matrix.iloc[int(markov_trace[i, t]), :].values)

    trace_df = pd.DataFrame(markov_trace).T
    trace_df['age_float'] = np.linspace(starting_age, starting_age + n_cycles * cycle_length, n_cycles + 1)
    trace_df['age'] = np.floor(trace_df['age_float'])
    trace_df.set_index(['age_float', 'age'], inplace=True)

    on_drug_df = pd.DataFrame(on_drug).T
    on_drug_df['age_float'] = np.linspace(starting_age, starting_age + n_cycles * cycle_length, n_cycles + 1)
    on_drug_df['age'] = np.floor(on_drug_df['age_float'])
    on_drug_df.set_index(['age_float', 'age'], inplace=True)

    return trace_df.T, on_drug_df.T


# TODO allow for more customizing cost and QALYs
def calculate_outcomes(markov_trace, cycle_length, treatment_type,
                       cost_suffix='', qaly_stage_suffix='', qaly_age_suffix='',
                       discount_rate=0.03, tx_cost_override=None, risk_attenuated_cycles=None,
                       begin_tx_at_cycle=0, drug_stages=None, on_drug=None):
    stage_costs, age_costs, treatment_cost = load_costs(os.path.join(DIR_HRP, "cost_input.csv"),
                                                        col_suffix=cost_suffix,
                                                        treatment_type=treatment_type,
                                                        tx_cost_override=tx_cost_override)
    stage_qalys, age_qalys, combined_qalys = load_qalys(os.path.join(DIR_HRP, "QALYs_input.csv"),
                                                        stage_suffix=qaly_stage_suffix,
                                                        age_suffix=qaly_age_suffix)

    cycle_tx_cost = treatment_cost * cycle_length

    # Cost is stage cost (matmul) + background age-related cost + treatment cost
    alive_stages = list(set(STAGE_ORDER) - set(MORTALITY_COLUMNS))
    cost_vector = markov_trace.loc[:, STAGE_ORDER] @ stage_costs.loc[STAGE_ORDER].values * cycle_length
    cost_vector += markov_trace.loc[:, alive_stages].sum(axis=1) * age_costs.loc[markov_trace['age']].values * cycle_length

    # only assume treatment for length of risk attenuation cycles (starting at begin_tx_at_cycle)
    # i.e. patient ceases treatment when efficacy wears off
    if risk_attenuated_cycles is None:
        risk_attenuated_cycles = len(markov_trace) - begin_tx_at_cycle
    if drug_stages is None:
        drug_stages = alive_stages

    if on_drug is None:
        on_drug = np.zeros(len(markov_trace))
        on_drug[begin_tx_at_cycle:begin_tx_at_cycle+risk_attenuated_cycles] = markov_trace.iloc[begin_tx_at_cycle:begin_tx_at_cycle+risk_attenuated_cycles][drug_stages].sum(axis=1)
    else:
        # if given MC trace for drug status, get proportion on drug at each cycle
        on_drug = (on_drug.sum() / on_drug.shape[0]).values
    cost_vector += on_drug * cycle_tx_cost

    # QALYs (age-specific)
    qaly_vector = np.sum(markov_trace.loc[:, STAGE_ORDER] * combined_qalys.loc[markov_trace['age'], STAGE_ORDER].values, axis=1) * cycle_length

    # calculate life-years, QALYs, and costs
    # ly_ones_death_zero = np.concat([np.ones(len(STAGE_ORDER) - 1), [0]])
    ly_ones_death_zero = np.append(np.ones(len(STAGE_ORDER) - 1), 0)

    ly_vector = markov_trace.loc[:, STAGE_ORDER] @ ly_ones_death_zero * cycle_length

    # make half cycle correction
    ly_vector_hcc = 0.5 * ly_vector[:-1].values + 0.5 * ly_vector[1:].values
    qaly_vector_hcc = 0.5 * qaly_vector[:-1].values + 0.5 * qaly_vector[1:].values
    cost_vector_hcc = 0.5 * cost_vector[:-1].values + 0.5 * cost_vector[1:].values

    print("Total LY: ", np.sum(ly_vector_hcc))
    print("Total QALY: ", np.sum(qaly_vector_hcc))
    print("Total Cost: ", np.sum(cost_vector_hcc))

    cycle_time = np.arange(0, len(ly_vector)) * cycle_length
    hcc_cycle_time = 0.5 * cycle_time[1:] + 0.5 * cycle_time[:-1]

    ly_vector_discount = discount(ly_vector_hcc, hcc_cycle_time, rate=discount_rate)
    qaly_vector_discount = discount(qaly_vector_hcc, hcc_cycle_time, rate=discount_rate)
    cost_vector_discount = discount(cost_vector_hcc, hcc_cycle_time, rate=discount_rate)

    print("Discounted LY: ", np.sum(ly_vector_discount))
    print("Discounted QALY: ", np.sum(qaly_vector_discount))
    print("Discounted Cost: ", np.sum(cost_vector_discount))

    return qaly_vector_discount, cost_vector_discount


def plot_trace(markov_df, line_type=None, legend=True):

    linestyle='solid' if line_type is None else line_type
    
    df = markov_df.drop(columns='age').set_index('age_float').stack().reset_index()
    df = df.rename(columns={'age_float': 'Age', 0: 'Proportion', 'level_1': 'Stage'})
    
    ax = sns.lineplot(data=df, x='Age', y='Proportion', hue='Stage', linestyle=linestyle, legend=legend)

    if line_type is not None and legend:
        main_legend = ax.legend(title="Stage", loc="upper left",
                               bbox_to_anchor=(1.02, 1))
 
        # Create a dummy line artist with your specific line_type to act as the legend key
        custom_line = Line2D([0], [0], color='black', linestyle='dotted')
        custom_line2 = Line2D([0], [0], color='black', linestyle='solid')
        custom_line3 = Line2D([0], [0], color='black', linestyle='dashed')
        
        # 3. Create the SECOND legend. 
        # Note: This will temporarily remove main_legend from the plot.
        # We assign it a different location (e.g., 'lower right') so they don't overlap.
        scenario_legend = ax.legend(
            handles=[custom_line, custom_line2, custom_line3], 
            labels=["SOC", "18", "12"], 
            title="Scenario", 
            loc="upper left",
            bbox_to_anchor=(1.02, 0.3)
        )
        
        # 4. Re-add the first legend back to the plot
        ax.add_artist(main_legend)

        plt.tight_layout()

        # Return the legends so we can pass them to savefig
        return main_legend, scenario_legend
    
    return None, None # Fallback if no legends are drawn


def run_single_model(tm_path, init_state, treatment_type,
                     rr=1, pr=1, n_cycles=None, starting_age=12, cycle_length=1/4,
                     transition_suffix='_obs',
                     risk_attenuated_cycles=None, risk_decreases=False, begin_tx_at_cycle=0,
                     plot=True, drug_stages=None, line_type=None, legend=True,
                     **outcomes_kwargs):
    print(cycle_length)

    if n_cycles is None:
        n_cycles = int((100 - starting_age) / cycle_length)

    # Prepare transition matrix
    transition_matrix = load_transition_matrix(tm_path, transition_suffix, cycle_length=cycle_length)

    # Get risk matrix
    risk_matrix = gen_risk_matrix(regression_risk=rr, progression_risk=pr)

    # Run Markov Trace and calculate outputs
    markov_trace = run_markov_cohort(transition_matrix, init_state, n_cycles, cycle_length,
                                     starting_age=starting_age, risk_matrix=risk_matrix,
                                     risk_attenuated_cycles=risk_attenuated_cycles, risk_decreases=risk_decreases,
                                     begin_tx_at_cycle=begin_tx_at_cycle, drug_stages=drug_stages)
    qaly_vector_discount, cost_vector_discount = calculate_outcomes(markov_trace, cycle_length, treatment_type,
                                                                    risk_attenuated_cycles=risk_attenuated_cycles,
                                                                    begin_tx_at_cycle=begin_tx_at_cycle,
                                                                    drug_stages=drug_stages,
                                                                    **outcomes_kwargs)
    # Plot results
    if plot:
        plot_trace(markov_trace, legend=legend, line_type=line_type)

    return qaly_vector_discount.sum(), cost_vector_discount.sum(), markov_trace


def run_comparison(tm_path, init_state, treatments, labels=None, rr=None, pr=None, **kwargs):
    if labels is None:
        labels = treatments

    kwargs_copy = kwargs.copy()

    results = {}
    traces = {}
    for i, (treatment, label) in enumerate(zip(treatments, labels)):
        print(f"Running model for treatment: {treatment}")
        rr_value = 1 if rr is None else rr[i]
        pr_value = 1 if pr is None else pr[i]

        # allow for passing in other changeable parameters
        for key, value in kwargs.items():
            if isinstance(value, list) and len(value) == len(treatments):
                kwargs_copy[key] = value[i]
        print(kwargs_copy)

        qaly, cost, trace = run_single_model(tm_path, init_state, treatment,
                                      rr=rr_value, pr=pr_value, **kwargs_copy)
        results[label] = {'QALY': qaly, 'Cost': cost}
        traces[label] = trace

    return results, traces