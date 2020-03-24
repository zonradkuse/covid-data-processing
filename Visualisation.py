import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display


class Visualisation:
    def __init__(self):
        self._component_output = widgets.Output()
        self._component_output.layout.width = 'calc(100% + 14ex)'
        self._component_output.layout.left = '-7ex'

        self._title = None
        self._description = None
        self._update = None

    def set_title(self, title):
        self._title = title

        return self

    def set_description(self, description):
        self._description = description

        return self

    def set_update_fn(self, fn):
        self._update = fn

        return self

    def vuetify_component(self):
        assert(self._title is not None)

        return v.Card(min_width="400px", max_width="500px", children=[
            v.CardTitle(children=[self._title]),
            v.CardText(children=[self._description, self._component_output])
        ])

    def update(self, countries):
        assert(self._update is not None)

        self._component_output.clear_output()
        with self._component_output:
            display(self._update(countries).get_figure())

    def vuetify_output(self):
        return self._component_output
