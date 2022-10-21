from typing import Any

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel
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
        self.loss = []

    def on_solution_callback(self):
        self.__solution_count += 1
        solution = list()
        for var_name, variables in self.__variables.items():
            for pos, variable in variables.items():
                if self.Value(variable):
                    print(f"{var_name}: ({pos[0]}, {pos[1]})")
                    solution.append((var_name, pos[0], pos[1]))
        print(f"Loss: {self.ObjectiveValue()}")
        self.all_solutions.append(solution)
        self.loss.append(self.ObjectiveValue())
        print()

    def solution_count(self):
        return self.__solution_count

    def plot_solutions(self):
        colors = ["#66c2ff", "#cc0000", "#ffc34d"]

        plot_df = pd.concat(
            [
                pd.DataFrame.from_records(
                    solution, columns=["name", "row", "column"]
                ).assign(solution_number=sol_number, loss=self.loss[sol_number])
                for sol_number, solution in enumerate(self.all_solutions)
            ]
        )
        if plot_df["loss"].sum() != 0:
            facet_by = "loss:O"
        else:
            facet_by = "solution_number:O"

        return (
            alt.Chart(plot_df)
            .mark_point(size=200, filled=True, opacity=1.0)
            .encode(
                x="column:O",
                y="row:O",
                shape=alt.Shape("name:N", sort=["◯", "□", "△"], legend=None),
                color=alt.Color(
                    "name:N",
                    scale=alt.Scale(domain=["◯", "□", "△"], range=colors),
                    legend=None,
                ),
            )
            .facet(facet_by, columns=8, background="#001325")
        )


# Define distance metric. Here the l1 distance
def distance(pos_1: tuple[int, int], pos_2: tuple[int, int]):
    return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])


def distance_to_living_quarter(position: tuple[int, int], max_column: int = 2) -> int:
    return abs(position[1] - max_column)


def add_distance_constraint(
    model: CpModel,
    x: dict[str, dict[tuple[int, int], Any]],
    name_1: str,
    name_2: str,
    min_distance: int,
    max_distance: int = 4,
) -> None:
    for pos_1, var_1 in x[name_1].items():
        for pos_2, var_2 in x[name_2].items():
            distance_var = model.NewIntVar(0, max_distance, "")
            model.Add(distance_var == distance(pos_1, pos_2))
            model.Add(distance_var > min_distance).OnlyEnforceIf([var_1, var_2])


def setup_and_optimize(with_preferences: bool) -> VarArraySolutionPrinter:
    # Creates the model.
    model = cp_model.CpModel()

    # Input for variables
    rows = range(0, 3)
    columns = range(0, 3)
    names = ["◯", "□", "△"]

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
        model.Add(sum([variable for variable in x[name].values()]) == 1)

    # Only one trailer per position
    for row in rows:
        for col in columns:
            model.Add(sum(x[name][(row, col)] for name in x) < 2)

    add_distance_constraint(model, x, "◯", "□", 2)
    add_distance_constraint(model, x, "□", "△", 2)
    add_distance_constraint(model, x, "△", "◯", 1)

    # Adding preferences (as far away from column 3)
    if with_preferences:
        loss = -1 * sum(
            distance_to_living_quarter(position, max(columns)) * variable
            for positions in x.values()
            for position, variable in positions.items()
        )

        # Minimize the loss
        model.Minimize(loss)

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
    default=False,
    help="Saves a figure of the solution",
)
def main(model_input_path, with_preferences: bool, save_figure: bool):
    solution_printer = setup_and_optimize(with_preferences)

    print(f"Number of solutions found: {solution_printer.solution_count()}")

    if save_figure:
        plot = solution_printer.plot_solutions()
        plot.save(model_input_path)


if __name__ == "__main__":
    main()
