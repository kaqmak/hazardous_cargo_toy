from itertools import combinations
from ortools.sat.python import cp_model

import pandas as pd
import altair as alt
import click

alt.themes.enable("dark")


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.all_solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        solution = list()
        for var_name, variables in self.__variables.items():
            for pos, variable in variables.items():
                if self.Value(variable):
                    print(f"{var_name}: ({pos[0]}, {pos[1]})")
                    solution.append((var_name, pos[0], pos[1]))
        self.all_solutions.append(solution)
        print()

    def solution_count(self):
        return self.__solution_count

    def plot_solutions(self):
        colors = ["#66c2ff", "#cc0000", "#ffc34d"]

        plot_df = pd.concat(
            [
                pd.DataFrame.from_records(
                    solution, columns=["name", "row", "column"]
                ).assign(solution_number=sol_number)
                for sol_number, solution in enumerate(self.all_solutions)
            ]
        )

        return (
            alt.Chart(plot_df)
            .mark_point(size=200, filled=True)
            .encode(
                x="row:O",
                y="column:O",
                shape="name:N",
                color=alt.Color("name:N", scale=alt.Scale(range=colors), legend=None),
            )
            .facet("solution_number", columns=4, background="#001325")
        )


def setup_and_optimize(with_preferences: bool) -> VarArraySolutionPrinter:
    # Creates the model.
    model = cp_model.CpModel()

    # Input for variables
    rows = range(0, 3)
    columns = range(0, 3)
    names = ["◯", "□", "△"]

    # Define distance metric. Here the l1 distance
    def l_1(pos_1: tuple[int, int], pos_2: tuple[int, int]):
        return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

    # Create variables
    x = {
        name: {
            (row, col): model.NewBoolVar(f"{name}_{row}_{col}")
            for col in columns
            for row in rows
        }
        for name in names
    }

    # Place trailer exactly once
    for name in names:
        model.Add(sum([variable for _, variable in x[name].items()]) == 1)

    # Only one trailer per position
    for row in rows:
        for col in columns:
            for name_1, name_2 in combinations(names, 2):
                var1 = x[name_1][(row, col)]
                var2 = x[name_2][(row, col)]
                model.AddBoolOr(
                    [var1.Not(), var2.Not()]  # not(var1 & var2)= not(var1) | not(var1)
                )

    # # distance betwen ◯ and □
    for pos_1, var_1 in x["◯"].items():
        for pos_2, var_2 in x["□"].items():
            distance_var = model.NewIntVar(0, max(rows) * max(columns), "")
            distance = l_1(pos_1, pos_2)
            model.Add(distance_var == distance)
            model.Add(distance_var > 2).OnlyEnforceIf([var_1, var_2])

    # # distance betwen □ and △
    for pos_1, var_1 in x["□"].items():
        for pos_2, var_2 in x["△"].items():
            distance_var = model.NewIntVar(0, max(rows) * max(columns), "")
            distance = l_1(pos_1, pos_2)
            model.Add(distance_var == distance)
            model.Add(distance_var > 2).OnlyEnforceIf([var_1, var_2])

    # # distance betwen □ and △
    for pos_1, var_1 in x["△"].items():
        for pos_2, var_2 in x["◯"].items():
            distance_var = model.NewIntVar(0, max(rows) * max(columns), "")
            distance = l_1(pos_1, pos_2)
            model.Add(distance_var == distance)
            model.Add(distance_var > 1).OnlyEnforceIf([var_1, var_2])

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solution_printer = VarArraySolutionPrinter(x)
    status = solver.Solve(model, solution_printer)
    if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        raise RuntimeError("Dammit! No solution was found.")

    return solution_printer


@click.command()
@click.argument("model_input_path", type=click.Path(), default="my_solution.svg")
@click.option(
    "--with_preferences",
    is_flag=True,
    show_default=True,
    default=False,
    help="Adds preferences",
)
@click.option(
    "--save_figure",
    is_flag=True,
    show_default=True,
    default=True,
    help="Plots and saves a figure of the solution",
)
def main(model_input_path, with_preferences: bool, save_figure: bool):
    solution_printer = setup_and_optimize(with_preferences)

    print(f"Number of solutions found: {solution_printer.solution_count()}")

    if save_figure:
        plot = solution_printer.plot_solutions()
        plot.save(model_input_path)


if __name__ == "__main__":
    main()
