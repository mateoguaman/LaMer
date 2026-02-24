"""
Hand-designed evaluation mazes of varying difficulty.
Each maze has a known optimal solution length for measuring efficiency.
"""

# 7x7 Easy maze - optimal: 6 steps
MAZE_7x7_EASY = """\
#######
#S....#
###.#.#
#...#.#
#.###.#
#.....#
####G##"""
MAZE_7x7_EASY_OPTIMAL = 6

# 7x7 Medium maze - optimal: 10 steps
MAZE_7x7_MEDIUM = """\
#######
#S.#..#
#.##..#
#.....#
###.#.#
#...#G#
#######"""
MAZE_7x7_MEDIUM_OPTIMAL = 10

# 9x9 Easy maze - optimal: 8 steps
MAZE_9x9_EASY = """\
#########
#S..#...#
#.#.#.#.#
#.#...#.#
#.#####.#
#.......#
#.###.###
#.....#G#
#########"""
MAZE_9x9_EASY_OPTIMAL = 8

# 9x9 Medium maze - optimal: 14 steps
MAZE_9x9_MEDIUM = """\
#########
#S#.....#
#.#.###.#
#.#.#...#
#.#.#.###
#...#...#
###.###.#
#.......#
######G##"""
MAZE_9x9_MEDIUM_OPTIMAL = 14

# 9x9 Hard maze - optimal: 16 steps
MAZE_9x9_HARD = """\
#########
#S..#...#
###.#.#.#
#...#.#.#
#.###.#.#
#.#...#.#
#.#.###.#
#.....#G#
#########"""
MAZE_9x9_HARD_OPTIMAL = 16

# 11x11 Medium maze - optimal: 18 steps
MAZE_11x11_MEDIUM = """\
###########
#S.#......#
#.##.####.#
#....#....#
####.#.##.#
#....#..#.#
#.####.##.#
#.#.......#
#.#.#####.#
#...#....G#
###########"""
MAZE_11x11_MEDIUM_OPTIMAL = 18

# 11x11 Hard maze - optimal: 24 steps
MAZE_11x11_HARD = """\
###########
#S#.......#
#.#.#####.#
#.#.#.....#
#.#.#.###.#
#...#...#.#
###.###.#.#
#.#...#.#.#
#.###.#.#.#
#.......#G#
###########"""
MAZE_11x11_HARD_OPTIMAL = 24

# Collection of all fixed mazes with metadata
FIXED_MAZES = [
    {"name": "7x7_easy", "maze": MAZE_7x7_EASY, "optimal": MAZE_7x7_EASY_OPTIMAL, "width": 7, "height": 7},
    {"name": "7x7_medium", "maze": MAZE_7x7_MEDIUM, "optimal": MAZE_7x7_MEDIUM_OPTIMAL, "width": 7, "height": 7},
    {"name": "9x9_easy", "maze": MAZE_9x9_EASY, "optimal": MAZE_9x9_EASY_OPTIMAL, "width": 9, "height": 9},
    {"name": "9x9_medium", "maze": MAZE_9x9_MEDIUM, "optimal": MAZE_9x9_MEDIUM_OPTIMAL, "width": 9, "height": 9},
    {"name": "9x9_hard", "maze": MAZE_9x9_HARD, "optimal": MAZE_9x9_HARD_OPTIMAL, "width": 9, "height": 9},
    {"name": "11x11_medium", "maze": MAZE_11x11_MEDIUM, "optimal": MAZE_11x11_MEDIUM_OPTIMAL, "width": 11, "height": 11},
    {"name": "11x11_hard", "maze": MAZE_11x11_HARD, "optimal": MAZE_11x11_HARD_OPTIMAL, "width": 11, "height": 11},
]
