from __future__ import annotations


from typing import (
    Any,
    Self
)

import attrs
import pyglet

from .event import Event
from .events import Events
from .toplevel import Toplevel


class WindowHandlers:
    __slots__ = ("__weakref__",)

    #def on_draw(
    #    self: Self
    #) -> None:
    #    Toplevel._get_window()._pyglet_window.clear()
    #    Toplevel._get_window()._batch.draw()

    def on_key_press(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.key_press(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_key_release(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.key_release(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_mouse_motion(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int
    ) -> None:
        Toplevel._get_window().push_event(Events.mouse_motion(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        ))

    def on_mouse_drag(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.mouse_drag(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_press(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.mouse_press(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_release(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.mouse_release(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_scroll(
        self: Self,
        x: int,
        y: int,
        scroll_x: float,
        scroll_y: float
    ) -> None:
        Toplevel._get_window().push_event(Events.mouse_scroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        ))

    def on_close(
        self: Self
    ) -> None:
        Toplevel._get_window().close()


@attrs.frozen(kw_only=True)
class WidgetsRecord:
    name_widget: pyglet.text.layout.ScrollableTextLayout
    status_widget: pyglet.text.layout.ScrollableTextLayout
    timer_widget: pyglet.text.layout.ScrollableTextLayout
    fps_widget: pyglet.text.layout.ScrollableTextLayout
    streaming_widget: pyglet.text.layout.ScrollableTextLayout
    recording_widget: pyglet.text.layout.ScrollableTextLayout
    log_widget: pyglet.text.layout.ScrollableTextLayout


class Window:
    __slots__ = (
        #"_window_handlers",
        #"_pyglet_window",
        "_pyglet_window",
        "_batch",
        #"_scene_canvas",
        "_widgets_record",
        "_window_size",
        "_scene_offset",
        "_scene_size",
        "_event_queue",
        "_window_handlers"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        scene_size = Toplevel._get_config().streaming_pixel_size
        batch = pyglet.graphics.Batch()
        if Toplevel._get_config().verbose:
            widgets_record, window_size, scene_offset = type(self)._get_verbose_window_info(
                scene_size=scene_size,
                font_name=Toplevel._get_config().window_font,
                batch=batch
            )
        else:
            widgets_record = None
            window_size = scene_size
            scene_offset = (0, 0)
            #scene_title_document = pyglet.text.document.UnformattedDocument()
            #scene_title_document.set_style(0, 0, {
            #    "font_name": font_name,
            #    "font_size": font_size,
            #    "bold": True
            #})
            #scene_title = pyglet.text.layout.TextLayout(
            #    scene_title_document,
            #    width=scene_width,
            #    height=scene_title_height,
            #    wrap_lines=False,
            #    batch=batch
            #)
            #scene_title.x = buff
            #scene_title.y = scene_box_height + 2 * buff + log_bar_height
            #scene_title.anchor_x = "left"
            #scene_title.anchor_y = "bottom"

        #scene_canvas = pyglet.shapes.Rectangle(
        #    x=scene_x,
        #    y=scene_y,
        #    width=scene_width,
        #    height=scene_height,
        #    color=(0, 0, 0, 0),
        #    batch=batch
        #)

        window_width, window_height = window_size
        major_version, minor_version = Toplevel._get_config().gl_version
        pyglet_window = pyglet.window.Window(
            width=window_width,
            height=window_height,
            config=pyglet.gl.Config(
                double_buffer=True,
                major_version=major_version,
                minor_version=minor_version
            )
        )
        window_handlers = WindowHandlers()
        pyglet_window.push_handlers(window_handlers)

        self._pyglet_window: pyglet.window.Window = pyglet_window
        self._batch: pyglet.graphics.Batch = batch
        #self._scene_canvas: pyglet.shapes.Rectangle = scene_canvas
        self._widgets_record: WidgetsRecord | None = widgets_record
        self._window_size: tuple[int, int] = window_size
        self._scene_offset: tuple[int, int] = scene_offset
        self._scene_size: tuple[int, int] = scene_size
        self._event_queue: list[Event] = []
        # Keep a strong reference to the handler object, as per
        # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
        self._window_handlers: WindowHandlers = window_handlers


        #batch = pyglet.graphics.Batch()
        #canvas = cls._get_pyglet_canvas(
        #    width=width,
        #    height=height,
        #    batch=batch
        #)
        #if preview:
        #    width, height = window_pixel_size
        #    major_version, minor_version = gl_version
        #    pyglet_window = PygletWindow(
        #        width=width,
        #        height=height,
        #        config=PygletWindowConfig(
        #            double_buffer=True,
        #            major_version=major_version,
        #            minor_version=minor_version
        #        )
        #    )
        #    window_handlers = WindowHandlers()
        #    pyglet_window.push_handlers(window_handlers)
        #else:
        #    window_handlers = None
        #    pyglet_window = None

        #self._pyglet_window: PygletWindow | None = pyglet_window

    #@property
    #def pyglet_window(
    #    self: Self
    #) -> PygletWindow:
    #    assert (pyglet_window := self._pyglet_window) is not None
    #    return pyglet_window

    @classmethod
    def _get_verbose_window_info(
        cls: type[Self],
        scene_size: tuple[int, int],
        font_name: str,
        batch: pyglet.graphics.Batch
    ) -> tuple[WidgetsRecord, tuple[int, int], tuple[int, int]]:

        def create_layout(
            *,
            batch: pyglet.graphics.Batch,
            x: int,
            y: int,
            width: int,
            height: int,
            attributes: dict[str, Any],
            #color: tuple[int, int, int, int] = (255, 255, 255, 255),
            text: str = "",
            multiline: bool = False
        ) -> pyglet.text.layout.ScrollableTextLayout:
            document = pyglet.text.document.UnformattedDocument()
            document.text = text
            document.set_style(0, len(text), attributes)
            layout = pyglet.text.layout.ScrollableTextLayout(
                document,
                width=width,
                height=height,
                multiline=multiline,
                wrap_lines=False,
                batch=batch
            )
            layout.x = x
            layout.y = y
            layout.anchor_x = "left"
            layout.anchor_y = "bottom"
            return layout

        scene_width, scene_height = scene_size
        scene_box_buff = 5
        scene_title_height = 30
        buff = 40
        title_height = 30
        info_value_height = 25
        info_item_height = 80
        info_bar_width = 120
        log_bar_height = 130
        scene_box_width = scene_width + 2 * scene_box_buff
        scene_box_height = scene_height + 2 * scene_box_buff
        info_x = scene_box_width + 2 * buff
        info_y = scene_box_height + 2 * buff + log_bar_height + scene_title_height

        #font_name = Toplevel._get_config().window_font
        font_size = 14
        log_font_size = 12
        color = (255, 255, 255, 255)

        pyglet.shapes.Box(
            x=buff,
            y=2 * buff + log_bar_height,
            width=scene_box_width,
            height=scene_box_height,
            batch=batch
        )
        name_widget = create_layout(
            batch=batch,
            x=buff,
            y=scene_box_height + 2 * buff + log_bar_height,
            width=scene_width,
            height=scene_title_height,
            attributes={
                "font_name": font_name,
                "font_size": font_size,
                "bold": True,
                "color": color
            }
        )
        _, (status_widget, timer_widget, fps_widget, streaming_widget, recording_widget) = zip(*(
            (
                create_layout(
                    batch=batch,
                    x=info_x,
                    y=info_y - index * info_item_height - title_height,
                    width=info_bar_width,
                    height=title_height,
                    attributes={
                        "font_name": font_name,
                        "font_size": font_size,
                        "bold": True,
                        "underline": color,
                        "color": color
                    },
                    text=info_title
                ),
                create_layout(
                    batch=batch,
                    x=info_x,
                    y=info_y - index * info_item_height - title_height - info_value_height,
                    width=info_bar_width,
                    height=info_value_height,
                    attributes={
                        "font_name": font_name,
                        "font_size": font_size,
                        "color": color
                    },
                    text=str(index)
                )
            )
            for index, info_title in enumerate((
                "Status",
                "Timer",
                "FPS",
                "Streaming",
                "Recording"
            ))
        ))
        create_layout(
            batch=batch,
            x=buff,
            y=buff + log_bar_height - title_height,
            width=scene_box_width + buff + info_bar_width,
            height=title_height,
            attributes={
                "font_name": font_name,
                "font_size": font_size,
                "bold": True,
                "underline": color,
                "color": color
            },
            text="Log Messages"
        )
        log_widget = create_layout(
            batch=batch,
            x=buff,
            y=buff,
            width=scene_box_width + buff + info_bar_width,
            height=log_bar_height - title_height,
            attributes={
                "font_name": font_name,
                "font_size": log_font_size,
                "color": color
            }
        )
        widgets_record = WidgetsRecord(
            name_widget=name_widget,
            status_widget=status_widget,
            timer_widget=timer_widget,
            fps_widget=fps_widget,
            streaming_widget=streaming_widget,
            recording_widget=recording_widget,
            log_widget=log_widget
        )
        window_size = (
            scene_box_width + 3 * buff + info_bar_width,
            scene_box_height + 3 * buff + log_bar_height + scene_title_height
        )
        scene_offset = (
            scene_box_buff + buff,
            scene_box_buff + 2 * buff + log_bar_height
        )
        return widgets_record, window_size, scene_offset

    def get_scene_viewport(
        self: Self
    ) -> tuple[int, int, int, int]:
        return (*self._scene_offset, *self._scene_size)

    def push_event(
        self: Self,
        event: Event
    ) -> None:
        self._event_queue.append(event)

    def capture_event(
        self: Self,
        target_event: Event
    ) -> Event | None:
        event_queue = self._event_queue
        for event in event_queue:
            if target_event._capture(event):
                event_queue.remove(event)
                return event
        return None

    def clear_event_queue(
        self: Self
    ) -> None:
        self._event_queue.clear()

    #def draw(
    #    self: Self
    #) -> None:
    #    self._pyglet_window.clear()
    #    self._batch.draw()
    #    if 

    def close(
        self: Self
    ) -> None:
        if (pyglet_window := self._pyglet_window) is not None:
            pyglet_window.close()
