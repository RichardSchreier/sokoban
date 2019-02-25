# Sokoban game in Python
With simple solver built on David Moreno's sokoban code.

## Screenshots
![Solved Image](https://raw.githubusercontent.com/RichardSchreier/sokoban/master/Solved.png "Solved")
![Unsolved Image](https://raw.githubusercontent.com/RichardSchreier/sokoban/master/Unsolved.png "Unsolved")

## Actions
```
Key     Action
←↑↓→    Move worker left, up, down, right
n/p     Next/previous level
N/P     Next/previous unsolved level
>/<     Next/previous world
q       Quit
r       Re-start
s       If no solution, solve from the current state.
S       Solve from the initial state, even if a solution has been found. While the solver is running:
            space   pause solve
            esc     exit solve
R       Replay solution
u/U     Undo/Re-do
mouse   Move worker to specified square; can push box adjacent to worker

Debug
^a      display annotated_map
^d      toggle DEBUG flag; DEBUG==True prints debug messages during solve()
^h      print heuristic
^m      print move_count_maps
^n      print neighbors
^s      print solution string
```

## Install & Play
1. Clone the repository `git clone https://github.com/RichardSchreier/sokoban.git`
2. Run `pip install -r requirements.txt`
3. Enjoy! `python3 sokoban.py`
