import click


def PromptIf(arg_name, arg_value):
    """ https://stackoverflow.com/a/49603426 """
    class Cls(click.Option):

        def __init__(self, *args, **kwargs):
            kwargs['prompt'] = kwargs.get('prompt', True)
            super(Cls, self).__init__(*args, **kwargs)

        def handle_parse_result(self, ctx, opts, args):
            assert any(c.name == arg_name for c in ctx.command.params), \
                "Param '{}' not found for option '{}'".format(
                    arg_name, self.name)

            if arg_name not in opts:
                raise click.UsageError(
                    "Illegal usage: `%s` is a required parameter with" % (
                        arg_name))

            # remove prompt from
            if opts[arg_name] != arg_value:
                self.prompt = None

            return super(Cls, self).handle_parse_result(ctx, opts, args)

    return Cls


class OptionPromptNull(click.Option):
    _value_key = '_default_val'

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