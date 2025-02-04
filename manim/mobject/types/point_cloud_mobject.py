"""Mobjects representing point clouds."""

from __future__ import annotations

__all__ = ["PMobject", "Mobject1D", "Mobject2D", "PGroup", "PointCloudDot", "Point"]

from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject

from ...constants import *
from ...mobject.mobject import Mobject
from ...utils.bezier import interpolate
from ...utils.color import (
    BLACK,
    WHITE,
    YELLOW,
    ManimColor,
    ManimColorList,
    ParsableManimColor,
    color_gradient,
    rgba_to_color,
    update_ColorList,
)
from ...utils.iterables import stretch_array_to_length

__all__ = ["PMobject", "Mobject1D", "Mobject2D", "PGroup", "PointCloudDot", "Point"]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy.typing as npt
    from typing_extensions import Self

    from manim.typing import ManimFloat, Point3DLike, Vector3D


class PMobject(Mobject, metaclass=ConvertToOpenGL):
    """A disc made of a cloud of Dots

    Examples
    --------

    .. manim:: PMobjectExample
        :save_last_frame:

        class PMobjectExample(Scene):
            def construct(self):

                pG = PGroup()  # This is just a collection of PMobject's

                # As the scale factor increases, the number of points
                # removed increases.
                for sf in range(1, 9 + 1):
                    p = PointCloudDot(density=20, radius=1).thin_out(sf)
                    # PointCloudDot is a type of PMobject
                    # and can therefore be added to a PGroup
                    pG.add(p)

                # This organizes all the shapes in a grid.
                pG.arrange_in_grid()

                self.add(pG)

    """

    __colors: ManimColorList = ManimColorList(None)
    # PMobject点集强制要求颜色列表与点集列表长度一致，但初始化时只有颜色没有点，所以只能额外用一个变量来存储
    _default_color: ManimColor = YELLOW

    def __init__(
        self,
        color: ParsableManimColor = YELLOW,
        stroke_width: int = DEFAULT_STROKE_WIDTH,
        **kwargs: Any,
    ) -> None:
        self.stroke_width = stroke_width
        super().__init__(**kwargs)

        self.colors = color

    def get_array_attrs(self) -> list[str]:
        return super().get_array_attrs() + ["colors"]

    def add_points(
        self,
        points: npt.NDArray,
        color: ParsableManimColor | None = None,
        opacity: float = 1.0,
    ) -> Self:
        """Add points.

        Points must be a Nx3 numpy array.
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if color is None:  # 如果不传入color，则使用默认颜色或最后一个颜色
            if len(self.colors) == 0:
                color = self._default_color
            else:
                color = self.colors[-1]
        color = ManimColorList(color, opacity)
        if len(color) == 1:
            color = np.repeat(color, len(points), axis=0)
        elif len(color) != len(points):
            raise ValueError("points and color must have same length")

        self.points = np.append(self.points, points, axis=0)
        self.__colors.append(color)
        return self

    def set_color(
        self,
        colors: ParsableManimColor = YELLOW,
        opacity: float = None,
        family: bool = True,
    ) -> Self:
        if family:
            for mob in self.family_members_with_points(include_self=False):
                mob.set_color(colors, opacity)
        if colors is None:
            self.__colors = update_ColorList(self.__colors, colors, opacity)
        else:  # 如果是单色，则应用到所有点颜色
            # ManimColor会自动识别是单颜色还是多颜色，单色返回ManimColor，多色返回ManimColorList
            if isinstance(ManimColor(colors, opacity), ManimColor):
                self._default_color = ManimColor(colors, opacity)
                self.__colors = update_ColorList(
                    self.__colors, colors, opacity, is_single_color=True
                )
            else:  # 如果是多色，则新旧颜色列表长度必须一样，因为PMobject要求颜色数量与点数量严格相等
                if len(colors) != len(self.points):
                    raise ValueError(
                        "In PMobject, points and colors must have same length. Please make sure your input 'colors' to have the same length as previous colors."
                    )
                self.__colors = update_ColorList(self.__colors, colors, opacity)

        return self

    def get_color(self) -> ManimColorList:
        # if len(self_colors != len(self.points):
        #     print("3333------------")
        #     print(self_colors len(self_colors)
        #     print(self.points, len(self.points))
        #     raise ValueError("points and color must have same length")
        return self.__colors

    colors = property(get_color, set_color)

    def get_stroke_width(self) -> int:
        return self.stroke_width

    def set_stroke_width(self, width: int, family: bool = True) -> Self:
        mobs = self.family_members_with_points() if family else [self]
        for mob in mobs:
            mob.stroke_width = width
        return self

    def set_color_by_gradient(self, *colors: ParsableManimColor) -> Self:
        self.colors = color_gradient(colors, len(self.points))
        return self

    def set_colors_by_radial_gradient(
        self,
        center: Point3DLike | None = None,
        radius: float = 1,
        inner_color: ParsableManimColor = WHITE,
        outer_color: ParsableManimColor = BLACK,
    ) -> Self:
        start_rgba = ManimColor(inner_color)
        end_rgba = ManimColor(outer_color)
        if center is None:
            center = self.get_center()
        for mob in self.family_members_with_points():
            distances = np.abs(self.points - center)
            alphas = np.linalg.norm(distances, axis=1) / radius

            mob.colors = ManimColorList(
                [interpolate(start_rgba, end_rgba, alpha) for alpha in alphas]
            )
        return self

    def match_colors(self, mobject: Mobject) -> Self:
        Mobject.align_data(self, mobject)
        self.colors = mobject.colors
        return self

    def filter_out(self, condition: npt.NDArray) -> Self:
        for mob in self.family_members_with_points():
            to_eliminate = ~np.apply_along_axis(condition, 1, mob.points)
            mob.points = mob.points[to_eliminate]
            mob.colors = mob.colors[to_eliminate]
        return self

    def thin_out(self, factor: int = 5) -> Self:
        """Removes all but every nth point for n = factor"""
        for mob in self.family_members_with_points():
            num_points = self.get_num_points()
            mob.apply_over_attr_arrays(
                lambda arr, n=num_points: arr[np.arange(0, n, factor)],
            )
        return self

    def sort_points(
        self, function: Callable[[npt.NDArray[ManimFloat]], float] = lambda p: p[0]
    ) -> Self:
        """Function is any map from R^3 to R"""
        for mob in self.family_members_with_points():
            indices = np.argsort(np.apply_along_axis(function, 1, mob.points))
            mob.apply_over_attr_arrays(lambda arr, idx=indices: arr[idx])
        return self

    def fade_to(
        self, color: ParsableManimColor, alpha: float, family: bool = True
    ) -> Self:
        self.colors = interpolate(self.colors, ManimColorList(color), alpha)
        for mob in self.submobjects:
            mob.fade_to(color, alpha, family)
        return self

    def get_all_rgbas(self) -> npt.NDArray:
        return self.get_merged_array("colors")

    def ingest_submobjects(self) -> Self:
        attrs = self.get_array_attrs()
        arrays = list(map(self.get_merged_array, attrs))
        for attr, array in zip(attrs, arrays):
            setattr(self, attr, array)
        self.submobjects = []
        return self

    def get_color(self) -> ManimColor:
        return rgba_to_color(self.colors[0, :])

    def point_from_proportion(self, alpha: float) -> Any:
        index = alpha * (self.get_num_points() - 1)
        return self.points[np.floor(index)]

    @staticmethod
    def get_mobject_type_class() -> type[PMobject]:
        return PMobject

    # Alignment
    def align_points_with_larger(self, larger_mobject: Mobject) -> None:
        assert isinstance(larger_mobject, PMobject)
        self.apply_over_attr_arrays(
            lambda a: stretch_array_to_length(a, larger_mobject.get_num_points()),
        )

    def get_point_mobject(self, center: Point3DLike | None = None) -> Point:
        if center is None:
            center = self.get_center()
        return Point(center)

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        self.colors = interpolate(mobject1.rgbas, mobject2.rgbas, alpha)
        self.set_stroke_width(
            interpolate(
                mobject1.get_stroke_width(),
                mobject2.get_stroke_width(),
                alpha,
            ),
        )
        return self

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> None:
        lower_index, upper_index = (int(x * mobject.get_num_points()) for x in (a, b))
        for attr in self.get_array_attrs():
            full_array = getattr(mobject, attr)
            partial_array = full_array[lower_index:upper_index]
            setattr(self, attr, partial_array)


# TODO, Make the two implementations below non-redundant
class Mobject1D(PMobject, metaclass=ConvertToOpenGL):
    def __init__(self, density: int = DEFAULT_POINT_DENSITY_1D, **kwargs: Any) -> None:
        self.density = density
        self.epsilon = 1.0 / self.density
        super().__init__(**kwargs)

    def add_line(
        self,
        start: npt.NDArray,
        end: npt.NDArray,
        color: ParsableManimColor | None = None,
    ) -> None:
        start, end = list(map(np.array, [start, end]))
        length = np.linalg.norm(end - start)
        if length == 0:
            points = np.array([start])
        else:
            epsilon = self.epsilon / length
            points = np.array(
                [interpolate(start, end, t) for t in np.arange(0, 1, epsilon)]
            )
        self.add_points(points, color=color)


class Mobject2D(PMobject, metaclass=ConvertToOpenGL):
    def __init__(self, density: int = DEFAULT_POINT_DENSITY_2D, **kwargs: Any) -> None:
        self.density = density
        self.epsilon = 1.0 / self.density
        super().__init__(**kwargs)


class PGroup(PMobject):
    """A group for several point mobjects.

    Examples
    --------

    .. manim:: PgroupExample
        :save_last_frame:

        class PgroupExample(Scene):
            def construct(self):

                p1 = PointCloudDot(radius=1, density=20, color=BLUE)
                p1.move_to(4.5 * LEFT)
                p2 = PointCloudDot()
                p3 = PointCloudDot(radius=1.5, stroke_width=2.5, color=PINK)
                p3.move_to(4.5 * RIGHT)
                pList = PGroup(p1, p2, p3)

                self.add(pList)

    """

    def __init__(self, *pmobs: Any, **kwargs: Any) -> None:
        if not all(isinstance(m, (PMobject, OpenGLPMobject)) for m in pmobs):
            raise ValueError(
                "All submobjects must be of type PMobject or OpenGLPMObject"
                " if using the opengl renderer",
            )
        super().__init__(**kwargs)
        self.add(*pmobs)

    def fade_to(
        self, color: ParsableManimColor, alpha: float, family: bool = True
    ) -> Self:
        if family:
            for mob in self.submobjects:
                mob.fade_to(color, alpha, family)
        return self


class PointCloudDot(Mobject1D):
    """A disc made of a cloud of dots.

    Examples
    --------
    .. manim:: PointCloudDotExample
        :save_last_frame:

        class PointCloudDotExample(Scene):
            def construct(self):
                cloud_1 = PointCloudDot(color=RED)
                cloud_2 = PointCloudDot(stroke_width=4, radius=1)
                cloud_3 = PointCloudDot(density=15)

                group = Group(cloud_1, cloud_2, cloud_3).arrange()
                self.add(group)

    .. manim:: PointCloudDotExample2

        class PointCloudDotExample2(Scene):
            def construct(self):
                plane = ComplexPlane()
                cloud = PointCloudDot(color=RED)
                self.add(
                    plane, cloud
                )
                self.wait()
                self.play(
                    cloud.animate.apply_complex_function(lambda z: np.exp(z))
                )
    """

    def __init__(
        self,
        center: Vector3D = ORIGIN,
        radius: float = 2.0,
        stroke_width: int = 2,
        density: int = DEFAULT_POINT_DENSITY_1D,
        color: ManimColor = YELLOW,
        **kwargs: Any,
    ) -> None:
        self.radius = radius
        self.epsilon = 1.0 / density
        super().__init__(
            stroke_width=stroke_width, density=density, color=color, **kwargs
        )
        self.shift(center)

    def generate_points(self) -> None:
        self.add_points(
            np.array(
                [
                    r * (np.cos(theta) * RIGHT + np.sin(theta) * UP)
                    for r in np.arange(self.epsilon, self.radius, self.epsilon)
                    # Num is equal to int(stop - start)/ (step + 1) reformulated.
                    for theta in np.linspace(
                        0,
                        2 * np.pi,
                        num=int(2 * np.pi * (r + self.epsilon) / self.epsilon),
                    )
                ]
            ),
        )

    init_points = generate_points


class Point(PMobject):
    """A mobject representing a point.

    Examples
    --------

    .. manim:: ExamplePoint
        :save_last_frame:

        class ExamplePoint(Scene):
            def construct(self):
                colorList = [RED, GREEN, BLUE, YELLOW]
                for i in range(200):
                    point = Point(location=[0.63 * np.random.randint(-4, 4), 0.37 * np.random.randint(-4, 4), 0], color=np.random.choice(colorList))
                    self.add(point)
                for i in range(200):
                    point = Point(location=[0.37 * np.random.randint(-4, 4), 0.63 * np.random.randint(-4, 4), 0], color=np.random.choice(colorList))
                    self.add(point)
                self.add(point)
    """

    def __init__(
        self, location: Vector3D = ORIGIN, color: ManimColor = BLACK, **kwargs: Any
    ) -> None:
        self.location = location
        super().__init__(color=color, **kwargs)

    def init_points(self) -> None:
        self.add_points(np.array([self.location]))
