import json

import sublime
import sublime_plugin


class ExportHighlightCommand(sublime_plugin.ApplicationCommand):
    def run(self):
        window = sublime.active_window()
        view = window.active_view()
        if view.is_loading():
            return

        full_region = sublime.Region(0, view.size())
        tokens = [
            {
                "begin": region.begin(),
                "end": region.end(),
                "style": view.style_for_scope(scope)
            }
            for region, scope in view.extract_tokens_with_scopes(full_region)
        ]
        stem, _ = view.file_name().rsplit(".", maxsplit=1)
        with open(stem + ".json", "w") as output_file:
            json.dump(tokens, output_file)

        view.close()
