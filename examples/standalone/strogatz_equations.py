"""
A selection of ordinary differential equations primarily from Steven Strogatz's book "Nonlinear Dynamics and Chaos" with manually chosen parameter values and initial conditions.
Some other famous known systems have been selected from other sources, which are included in the dictionary entries as well.
We selected ODEs primarily based on whether they have actually been suggested as models for real-world phenomena as well as on whether they are 'iconic' ODEs in the sense that they are often used as examples in textbooks and/or have recognizable names.
Whenever there were 'realistic' parameter values suggested, we chose those.
In this benchmark, we typically include only one set of parameter values per equation.
Many of the ODEs in Strogatz' book are analyzed in terms of the different limiting behavior for different parameter settings.
For some systems that exhibit wildely different behavior for different parameter settings, we include multiple sets of parameter values as separate equations (e.g., Lorenz system in chaotic and non-chaotic regime).
For each equation, we include two sets of manually chosen initial conditions.
"""


equations = [
{
    'id': 1,
    'eq': '(c_0 - x_0 / c_1) / c_2',
    'eq_gt': 'c_0 * c_1 + (i_0 - c_0 * c_1) * exp(- t / (c_1 * c_2))',
    'mnn': ([[['1. / c_1', 'c_2']]], ['c_0']),
    'dim': 1,
    'consts': [[0.7, 1.2, 2.31]],
    'init': [[10.], [3.54]],
    'init_constraints': 'x_0 > 0',
    'const_constraints': 'c_1 > 0, c_2 > 0',
    'eq_description': 'RC-circuit (charging capacitor)',
    'const_description': 'c_0: fixed voltage source, c_1: capacitance, c_2: resistance',
    'var_description': 'x_0: charge',
    'source': 'strogatz p.20'
},
{
    'id': 2,
    'eq': 'c_0 * x_0',
    'eq_gt': 'i_0 * exp(c_0 * t)',
    'mnn': ([[['c_0', '-1.']]], ['0.']),
    'dim': 1,
    'consts': [[0.23]],
    'init': [[4.78], [0.87]],
    'init_constraints': 'x_0 > 0',
    'const_constraints': '',
    'eq_description': 'Population growth (naive)',
    'const_description': 'c_0: growth rate',
    'var_description': 'x_0: population',
    'source': 'strogatz p.22'
},
{
    'id': 9,
    'eq': '(1 - x_0) * c_0 - x_0 * c_1',
    'eq_gt': 'c_0 / (c_0 + c_1) - (c_0 / (c_0 + c_1) - i_0) * exp(- t * (c_0 + c_1))',
    'mnn': ([[['c_0 + c_1', '1.']]], ['c_0']),
    'dim': 1,
    'consts': [[0.32, 0.28]],
    'init': [[0.14], [0.55]],
    'init_constraints': '0 < x_0 < 1',
    'const_constraints': 'c_0 >= 0, c_1 >= 0',
    'eq_description': 'Language death model for two languages',
    'const_description': 'c_0: rate of language 1 speakers switching to language 2, c_1: rate of language 2 speakers switching to language 1',
    'var_description': 'x_0: proportion of population speaking language 1',
    'source': 'strogatz p.40'
},
{
    'id': 24,
    'eq': 'x_1 | - c_0 * x_0',
    'eq_gt': 'i_0 * cos(sqrt(c_0) * t) + i_1 / sqrt(c_0) * sin(sqrt(c_0) * t) | '
             'i_1 * cos(sqrt(c_0) * t) - sqrt(c_0) * i_0 * sin(sqrt(c_0) * t)',
    'mnn': ([[['0.', '1.'], ['-1.', '0.']], [['c_0', '0.'], ['0.', '1.']]], ['0.', '0.']),
    'dim': 2,
    'consts': [[2.1]],
    'init': [[0.4, -0.03], [0.0, 0.2]],
    'init_constraints': '',
    'const_constraints': 'c_0 > 0',
    'eq_description': 'Harmonic oscillator without damping',
    'const_description': 'c_0: spring constant to mass ratio',
    'var_description': 'x_0: position, x_1: velocity',
    'source': 'strogatz p.126'
},
{
    'id': 25,
    'eq': 'x_1 | - c_0 * x_0 - c_1 * x_1',
    'eq_gt': 'exp(-.5 * c_1 * t) * (i_0 * cos(sqrt(c_0 - .25 * c_1 ^ 2) * t) + (.5 * c_1 * i_0 + i_1) / sqrt(c_0 - .25 * c_1 ^ 2) * sin(sqrt(c_0 - .25 * c_1 ^ 2) * t)) | '
             'exp(-.5 * c_1 * t) * (i_1 * cos(sqrt(c_0 - .25 * c_1 ^ 2) * t) - (.5 * c_1 * (.5 * c_1 * i_0 + i_1) / sqrt(c_0 - .25 * c_1 ^ 2) + sqrt(c_0 - .25 * c_1 ^ 2) * i_0) * sin(sqrt(c_0 - .25 * c_1 ^ 2) * t))',
    'mnn': ([[['0.', '1.'], ['-1.', '0.']], [['c_0', '0.'], ['c_1', '1.']]], ['0.', '0.']),
    'dim': 2,
    'consts': [[4.5, 0.43]],
    'init': [[0.12, 0.043], [0.0, -0.3]],
    'init_constraints': '',
    'const_constraints': 'c_0 > 0, c_1 > 0',
    'eq_description': 'Harmonic oscillator with damping',
    'const_description': 'c_0: spring constant to mass ratio, c_1: damping coefficient to mass ratio',
    'var_description': 'x_0: position, x_1: velocity',
    'source': 'strogatz p.144'
},
{
    'id': 99,
    'eq': 'x_1 | x_2 | - x_1 - x_2',
    'eq_gt': 'i_0 + i_1 + i_2 + exp(-.5 * t) * (-(i_1 + i_2) * cos(sqrt(.75) * t) + (i_1 - i_2) / sqrt(3.) * sin(sqrt(.75) * t)) | '
             'exp(-.5 * t) * (i_1 * cos(sqrt(.75) * t) + (i_1 + 2. * i_2) / sqrt(3.) * sin(sqrt(.75) * t)) | '
             'exp(-.5 * t) * (i_2 * cos(sqrt(.75) * t) - (2. * i_1 + i_2) / sqrt(3.) * sin(sqrt(.75) * t))',
    # next: 'exp(-.5 * t) * (-(i_1 + i_2) * cos(sqrt(.75) * t) + (i_1 - i_2) / sqrt(3.) * sin(sqrt(.75) * t))'
    'mnn': ([[['0.', '1.'], ['-1.', '0.'], ['0.', '0.']], [['0.', '0.'], ['0.', '1.'], ['-1.', '0.']], [['0.', '0.'], ['1.', '0.'], ['1.', '1.']]], ['0.', '0.', '0.']),
    'dim': 3,
    'consts': [[]],
    'init': [[0., -1., 1.]],
    'eq_description': 'third-order',
},
]
