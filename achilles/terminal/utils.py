import click
import json

from pathlib import Path


class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def PromptIf(arg_name, arg_value):
    """ https://stackoverflow.com/a/49603426 """

    class Cls(click.Option):
        def __init__(self, *args, **kwargs):
            kwargs["prompt"] = kwargs.get("prompt", True)
            super(Cls, self).__init__(*args, **kwargs)

        def handle_parse_result(self, ctx, opts, args):
            assert any(
                c.name == arg_name for c in ctx.command.params
            ), "Param '{}' not found for option '{}'".format(arg_name, self.name)

            if arg_name not in opts:
                raise click.UsageError(
                    "Illegal usage: `%s` is a required parameter with" % (arg_name)
                )

            # remove prompt from
            if opts[arg_name] != arg_value:
                self.prompt = None

            return super(Cls, self).handle_parse_result(ctx, opts, args)

    return Cls


class OptionPromptNull(click.Option):
    _value_key = "_default_val"

    def get_default(self, ctx):
        if not hasattr(self, self._value_key):
            default = super(OptionPromptNull, self).get_default(ctx)
            setattr(self, self._value_key, default)
        return getattr(self, self._value_key)

    def prompt_for_value(self, ctx):
        default = self.get_default(ctx)

        # only prompt if the default value is None
        if default is None:
            return super(OptionPromptNull, self).prompt_for_value(ctx), True

        return default


def get_uri(pmid):

    config = read_config_path(config_file="poremongo.json")

    try:
        return config[pmid]
    except KeyError:
        click.echo(f"Could not find PMID: {pmid}")


def read_config_path(config_file="config.json") -> dict:

    config_path = Path.home() / ".achilles" / config_file

    if config_path.exists():
        with open(config_path.absolute(), "r") as json_file:
            return json.load(json_file)
    else:
        return dict()


def write_config_path(config, config_file="config.json"):

    achilles_path = Path.home() / ".achilles"

    achilles_path.mkdir(parents=True, exist_ok=True)

    config_path = achilles_path / config_file

    with open(config_path.absolute(), "w") as json_file:
        return json.dump(config, json_file)
