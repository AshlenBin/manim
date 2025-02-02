"""Manim's (internal) color data structure and some utilities for color conversion.

This module contains the implementation of :class:`.ManimColor`, the data structure
internally used to represent colors.

The preferred way of using these colors is by importing their constants from Manim:

.. code-block:: pycon

    >>> from manim import RED, GREEN, BLUE
    >>> print(RED)
    #FC6255

Note that this way uses the name of the colors in UPPERCASE.

.. note::

    The colors with a ``_C`` suffix have an alias equal to the colorname without a
    letter. For example, ``GREEN = GREEN_C``.

===================
Custom Color Spaces
===================

Hello, dear visitor. You seem to be interested in implementing a custom color class for
a color space we don't currently support.

The current system is using a few indirections for ensuring a consistent behavior with
all other color types in Manim.

To implement a custom color space, you must subclass :class:`ManimColor` and implement
three important methods:

  - :attr:`~.ManimColor._internal_value`: a ``@property`` implemented on
    :class:`ManimColor` with the goal of keeping a consistent internal representation
    which can be referenced by other functions in :class:`ManimColor`. This property acts
    as a proxy to whatever representation you need in your class.

      - The getter should always return a NumPy array in the format ``[r,g,b,a]``, in
        accordance with the type :class:`ManimColorInternal`.

      - The setter should always accept a value in the format ``[r,g,b,a]`` which can be
        converted to whatever attributes you need.

  - :meth:`~ManimColor._from_internal`: a ``@classmethod`` which converts an
    ``[r,g,b,a]`` value into suitable parameters for your ``__init__`` method and calls
    the ``cls`` parameter.
"""

from __future__ import annotations

import colorsys

# logger = _config.logger
import random
import re
from collections.abc import Sequence
from typing import TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeAlias, TypeIs, override

from manim.typing import (
    HSL_Array_Float,
    HSL_Tuple_Float,
    HSV_Array_Float,
    HSV_Tuple_Float,
    HSVA_Array_Float,
    HSVA_Tuple_Float,
    ManimColorDType,
    ManimColorInternal,
    ManimFloat,
    Point3D,
    RGB_Array_Float,
    RGB_Array_Int,
    RGB_Tuple_Float,
    RGB_Tuple_Int,
    RGBA_Array_Float,
    RGBA_Array_Int,
    RGBA_Tuple_Float,
    RGBA_Tuple_Int,
    Vector3D,
)

from ...utils.space_ops import normalize

# import manim._config as _config

re_hex = re.compile("((?<=#)|(?<=0x))[A-F0-9]{3,8}", re.IGNORECASE)


class ManimColor(np.ndarray):
    """A color class for Manim.
    Parameters
    ----------
    value
        Some representation of a color (e.g., a string or
        a suitable tuple). The default ``None`` is ``BLACK``.
    alpha
        The opacity of the color. By default, colors are
        fully opaque (value 1.0).
        如果传入alpha不为None，则将其作为不透明度，否则不透明度默认为1.0
         如果value是RGBa，则alpha覆盖RGBa的a
        比如：ManimColor((23,24,25,0.8), 0.5) , 则ManimColor的不透明度为0.5而不是0.8
        【注意】不接受单独数字的灰度值！如果你想输入灰度值列表，请直接使用ManimColor.gray([1,2,3,4,5])
    """

    def __new__(
        cls,
        value: ParsableManimColor | None = None,
        alpha: float | None = None,
        scale: int = 255,
    ) -> None:
        if (
            isinstance(value, (Sequence, np.ndarray))
            and not isinstance(value, str)
            and not all(isinstance(c, (int, float)) for c in value)
        ):
            """自动识别输入的是“颜色”还是“颜色列表”
            输入：color= [1,2,3] 视为RGB
            返回：ManimColor，RGBa=[1/255,2/255,3/255,1]
            输入：color= [1,2,3,4] 视为RGBa，a=4越界自动截断
            返回：ManimColor，RGBa=[1/255,2/255,3/255,1]
            输入：color= [[1,2,3],[4,5,6]] 视为两个RGB值
            返回：ManimColorList，[[1/255,2/255,3/255,1],[4/255,5/255,6/255,1]]
            输入：color= [1,2,3,4,5] 视为RGBa，
            返回：报错，ManimColor初始化发现color长度不为3或4
            如果你想输入灰度值列表，请直接使用ManimColor.gray([1,2,3,4,5])
            """
            return ManimColorList(value, alpha)

        _internal_value = np.array((0, 0, 0, 1), dtype=ManimColorDType)

        if value is None:
            pass
        elif isinstance(value, ManimColor):
            _internal_value = value.copy()
        elif isinstance(value, int):  # 十六进制数，不是灰度值！
            _internal_value[0] = ((value >> 16) & 0xFF) / 255
            _internal_value[1] = ((value >> 8) & 0xFF) / 255
            _internal_value[2] = ((value >> 0) & 0xFF) / 255

        elif isinstance(value, str):  # hex十六进制颜色值 或 颜色名称
            result = re_hex.search(value)
            if result is not None:  # hex颜色值
                _internal_value = cls._internal_from_hex_string(result.group())
            else:  # 颜色名称
                # This is not expected to be called on module initialization time
                # It can be horribly slow to convert a string to a color because
                # it has to access the dictionary of colors and find the right color
                _internal_value = cls._internal_from_string(value)
        elif isinstance(value, (np.ndarray, Sequence)):
            length = len(value)
            if length != 3 and length != 4:
                raise ValueError(
                    f"ManimColor only accepts lists/tuples/arrays of length 3 or 4. You input: value = {value}"
                )
            if length == 4:  # RGBa，不透明度a取值范围应当在0-1之间，如果越界则自动调整
                if value[3] < 0:
                    value[3] = 0
                elif value[3] > 1:
                    value[3] = 1
            for i in range(3):
                if value[i] < 0:
                    value[i] = 0
                elif value[i] > scale:
                    value[i] = scale
            for i in range(length):  # 传参value写入_internal_value
                _internal_value[i] = value[i]
            _internal_value[:3] /= scale  # RGB取值范围转换为[0~1]
        else:
            # logger.error(f"Invalid color value: {value}")
            raise TypeError(
                "ManimColor only accepts int, str, list[int, int, int], "
                "list[int, int, int, int], list[float, float, float], "
                f"list[float, float, float, float]. You input: value =  {value}"
            )
        if alpha is not None:
            # 如果传入alpha不为None，则将其作为不透明度，否则不透明度默认为1.0
            # 比如如果value是RGBa，则alpha覆盖RGBa的a
            _internal_value[3] = alpha
        return _internal_value.view(cls).copy()

    def __array_finalize__(self, obj: Self) -> None:
        # 【功能】[0~1]越界截断
        if obj is None:
            return
        if self.dtype == bool:
            # self<0花式索引会创建新的ManimColor对象，导致无限递归
            return

        self[self < 0] = 0
        self[self > 1] = 1

    def gray(value: int | Sequence[int] | np.ndarray[int], opacity: float = 1.0):
        """Create a gray color with the given value.
        输入：value灰度值[0~255]，opacity不透明度
        输出：ManimColor对象，灰度值转为RGB
        若value为序列，则返回ManimColorList
        """
        if value is None:
            raise ValueError("'value' cannot be None")
        if isinstance(value, int):
            return ManimColor([value, value, value, opacity])
        if isinstance(value, (np.ndarray, Sequence)):
            return ManimColorList([(v, v, v, opacity) for v in value])

    @staticmethod
    def _internal_from_hex_string(hex_: str) -> ManimColorInternal:
        """Internal function for converting a hex string into the internal representation
        of a :class:`ManimColor`.

        .. warning::
            This does not accept any prefixes like # or similar in front of the hex string.
            This is just intended for the raw hex part.

        *For internal use only*

        Parameters
        ----------
        hex
            Hex string to be parsed.

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
        alpha = 1.0
        if len(hex_) in (3, 4):
            hex_ = "".join([x * 2 for x in hex_])
        if len(hex_) == 6:
            hex_ += "FF"
        elif len(hex_) == 8:
            alpha = (int(hex_, 16) & 0xFF) / 255
        else:
            raise ValueError(
                "Hex colors must be specified with either 0x or # as prefix and contain 6 or 8 hexadecimal numbers"
            )
        tmp = int(hex_, 16)
        return np.array(
            (
                ((tmp >> 24) & 0xFF) / 255,
                ((tmp >> 16) & 0xFF) / 255,
                ((tmp >> 8) & 0xFF) / 255,
                alpha,
            ),
            dtype=ManimColorDType,
        )

    @staticmethod
    def _internal_from_string(name: str) -> ManimColorInternal:
        """Internal function for converting a string into the internal representation of
        a :class:`ManimColor`. This is not used for hex strings: please refer to
        :meth:`_internal_from_hex` for this functionality.

        *For internal use only*

        Parameters
        ----------
        name
            The color name to be parsed into a color. Refer to the different color
            modules in the documentation page to find the corresponding color names.

        Returns
        -------
        ManimColorInternal
            Internal color representation.

        Raises
        ------
        ValueError
            If the color name is not present in Manim.
        """
        from . import _all_color_dict

        if tmp := _all_color_dict.get(name.upper()):
            tmp[3] = 1
            return tmp.copy()
        else:
            raise ValueError(f"Color {name} not found")

    def to_integer(self) -> int:
        """Convert the current :class:`ManimColor` into an integer.

        .. warning::
            This will return only the RGB part of the color.

        Returns
        -------
        int
            Integer representation of the color.
        """
        tmp = (self[:3] * 255).astype(dtype=np.byte).tobytes()
        return int.from_bytes(tmp, "big")

    def to_rgb(self) -> RGB_Array_Float:
        """Convert the current :class:`ManimColor` into an RGB array of floats.

        Returns
        -------
        RGB_Array_Float
            RGB array of 3 floats from 0.0 to 1.0.
        """
        return self[:3]

    def to_int_rgb(self) -> RGB_Array_Int:
        """Convert the current :class:`ManimColor` into an RGB array of integers.

        Returns
        -------
        RGB_Array_Int
            RGB array of 3 integers from 0 to 255.
        """
        return (self[:3] * 255).astype(int)

    def to_rgba(self) -> RGBA_Array_Float:
        """Convert the current :class:`ManimColor` into an RGBA array of floats.

        Returns
        -------
        RGBA_Array_Float
            RGBA array of 4 floats from 0.0 to 1.0.
        """
        return self

    def to_int_rgba(self) -> RGBA_Array_Int:
        """Convert the current ManimColor into an RGBA array of integers.


        Returns
        -------
        RGBA_Array_Int
            RGBA array of 4 integers from 0 to 255.
        """
        return (self * 255).astype(int)

    def to_hex(self, with_alpha: bool = False) -> str:
        """Convert the :class:`ManimColor` to a hexadecimal representation of the color.

        Parameters
        ----------
        with_alpha
            If ``True``, append 2 extra characters to the hex string which represent the
            alpha value of the color between 0 and 255. Default is ``False``.

        Returns
        -------
        str
            A hex string starting with a ``#``, with either 6 or 8 nibbles depending on
            the ``with_alpha`` parameter. By default, it has 6 nibbles, i.e. ``#XXXXXX``.
        """
        tmp = (
            f"#{int(self[0] * 255):02X}"
            f"{int(self[1] * 255):02X}"
            f"{int(self[2] * 255):02X}"
        )
        if with_alpha:
            tmp += f"{int(self[3] * 255):02X}"
        return tmp

    def to_hsv(self) -> HSV_Array_Float:
        """Convert the :class:`ManimColor` to an HSV array.

        .. note::
           Be careful: this returns an array in the form ``[h, s, v]``, where the
           elements are floats. This might be confusing, because RGB can also be an array
           of floats. You might want to annotate the usage of this function in your code
           by typing your HSV array variables as :class:`HSV_Array_Float` in order to
           differentiate them from RGB arrays.

        Returns
        -------
        HSV_Array_Float
            An HSV array of 3 floats from 0.0 to 1.0.
        """
        return np.array(colorsys.rgb_to_hsv(*self.to_rgb()))

    def to_hsl(self) -> HSL_Array_Float:
        """Convert the :class:`ManimColor` to an HSL array.

        .. note::
           Be careful: this returns an array in the form ``[h, s, l]``, where the
           elements are floats. This might be confusing, because RGB can also be an array
           of floats. You might want to annotate the usage of this function in your code
           by typing your HSL array variables as :class:`HSL_Array_Float` in order to
           differentiate them from RGB arrays.

        Returns
        -------
        HSL_Array_Float
            An HSL array of 3 floats from 0.0 to 1.0.
        """
        return np.array(colorsys.rgb_to_hls(*self.to_rgb()))

    def invert(self, with_alpha: bool = False) -> Self:
        """Return a new, linearly inverted version of this :class:`ManimColor` (no
        inplace changes).

        Parameters
        ----------
        with_alpha
            If ``True``, the alpha value will be inverted too. Default is ``False``.

            .. note::
                Setting ``with_alpha=True`` can result in unintended behavior where
                objects are not displayed because their new alpha value is suddenly 0 or
                very low.

        Returns
        -------
        ManimColor
            The linearly inverted :class:`ManimColor`.
        """
        if with_alpha:
            return np.subtract(1, self)
        else:
            alpha = self[3]
            new = np.subtract(1, self)
            new[-1] = alpha
            return new

    def interpolate(self, other: Self, weight: float) -> Self:
        """Interpolate between the current and the given :class:`ManimColor`, and return
        the result.

        Parameters
        ----------
        other
            The other :class:`ManimColor` to be used for interpolation.
        alpha
            A point on the line in RGBA colorspace connecting the two colors, i.e. the
            interpolation point. 0.0 corresponds to the current :class:`ManimColor` and
            1.0 corresponds to the other :class:`ManimColor`.

        Returns
        -------
        ManimColor
            The interpolated :class:`ManimColor`.
        """
        return self * (1 - weight) + other * weight

    def darker(self, blend: float = 0.2) -> Self:
        """Return a new color that is darker than the current color, i.e.
        interpolated with ``BLACK``. The opacity is unchanged.

        Parameters
        ----------
        blend
            The blend ratio for the interpolation, from 0.0 (the current color
            unchanged) to 1.0 (pure black). Default is 0.2, which results in a
            slightly darker color.

        Returns
        -------
        ManimColor
            The darker :class:`ManimColor`.

        See Also
        --------
        :meth:`lighter`
        """
        from manim.utils.color.manim_colors import BLACK

        alpha = self[3]
        result = self.interpolate(BLACK, blend)
        result[3] = alpha
        return result

    def lighter(self, blend: float = 0.2) -> Self:
        """Return a new color that is lighter than the current color, i.e.
        interpolated with ``WHITE``. The opacity is unchanged.

        Parameters
        ----------
        blend
            The blend ratio for the interpolation, from 0.0 (the current color
            unchanged) to 1.0 (pure white). Default is 0.2, which results in a
            slightly lighter color.

        Returns
        -------
        ManimColor
            The lighter :class:`ManimColor`.

        See Also
        --------
        :meth:`darker`
        """
        from manim.utils.color.manim_colors import WHITE

        alpha = self[3]
        result = self.interpolate(WHITE, blend)
        result[3] = alpha
        return result

    def contrasting(
        self,
        threshold: float = 0.5,
        light: Self | None = None,
        dark: Self | None = None,
    ) -> Self:
        """Return one of two colors, light or dark (by default white or black),
        that contrasts with the current color (depending on its luminance).
        This is typically used to set text in a contrasting color that ensures
        it is readable against a background of the current color.

        Parameters
        ----------
        threshold
            The luminance threshold which dictates whether the current color is
            considered light or dark (and thus whether to return the dark or
            light color, respectively). Default is 0.5.
        light
            The light color to return if the current color is considered dark.
            Default is ``None``: in this case, pure ``WHITE`` will be returned.
        dark
            The dark color to return if the current color is considered light,
            Default is ``None``: in this case, pure ``BLACK`` will be returned.

        Returns
        -------
        ManimColor
            The contrasting :class:`ManimColor`.
        """
        from manim.utils.color.manim_colors import BLACK, WHITE

        luminance, _, _ = colorsys.rgb_to_yiq(*self.to_rgb())
        if luminance < threshold:
            if light is not None:
                return light
            return WHITE
        else:
            if dark is not None:
                return dark
            return BLACK

    @property
    def opacity(self) -> float:
        """获取颜色的不透明度"""
        return self[3]

    def into(self, class_type: type[ManimColorT]) -> ManimColorT:
        """Convert the current color into a different colorspace given by ``class_type``,
        without changing the :attr:`_internal_value`.

        Parameters
        ----------
        class_type
            The class that is used for conversion. It must be a subclass of
            :class:`ManimColor` which respects the specification HSV, RGBA, ...

        Returns
        -------
        ManimColorT
            A new color object of type ``class_type`` and the same
            :attr:`_internal_value` as the original color.
        """
        return class_type(self, scale=1)

    @classmethod
    def from_hsv(
        cls, hsv: HSV_Array_Float | HSV_Tuple_Float, alpha: float = 1.0
    ) -> Self:
        """Create a :class:`ManimColor` from an HSV array.

        Parameters
        ----------
        hsv
            Any iterable containing 3 floats from 0.0 to 1.0.
        alpha
            The alpha value to be used. Default is 1.0.

        Returns
        -------
        ManimColor
            The :class:`ManimColor` with the corresponding RGB values to the given HSV
            array.
        """
        rgb = colorsys.hsv_to_rgb(*hsv)

        return ManimColor(rgb, alpha, scale=1)

    @classmethod
    def from_hsl(
        cls, hsl: HSL_Array_Float | HSL_Tuple_Float, alpha: float = 1.0
    ) -> Self:
        """Create a :class:`ManimColor` from an HSL array.

        Parameters
        ----------
        hsl
            Any iterable containing 3 floats from 0.0 to 1.0.
        alpha
            The alpha value to be used. Default is 1.0.

        Returns
        -------
        ManimColor
            The :class:`ManimColor` with the corresponding RGB values to the given HSL
            array.
        """
        rgb = colorsys.hls_to_rgb(*hsl)
        return ManimColor(rgb, alpha, scale=1)

    @overload
    @classmethod
    def parse(
        cls,
        color: ParsableManimColor | None,
        alpha: float = ...,
    ) -> Self: ...

    @overload
    @classmethod
    def parse(
        cls,
        color: Sequence[ParsableManimColor],
        alpha: float = ...,
    ) -> list[Self]: ...

    @classmethod
    def parse(
        cls,
        color: ParsableManimColor | Sequence[ParsableManimColor],
        alpha: float | None = None,
    ) -> Self | list[Self]:
        """自动识别输入的是“颜色”还是“颜色列表”
        输入：color= [1,2,3] 视为RGB
         返回：ManimColor，RGBa=[1/255,2/255,3/255,1]
        输入：color= [1,2,3,4] 视为RGBa，a=4越界自动截断
         返回：ManimColor，RGBa=[1/255,2/255,3/255,1]
        输入：color= [[1,2,3],[4,5,6]] 视为两个RGB值
         返回：ManimColorList，[[1/255,2/255,3/255,1],[4/255,5/255,6/255,1]]
        输入：color= [1,2,3,4,5] 视为RGBa，
         返回：报错，ManimColor初始化发现color长度不为3或4
        如果你想输入灰度值列表，请直接使用ManimColor.gray([1,2,3,4,5])
        """

        if isinstance(color, (Sequence, np.ndarray)) and not all(
            isinstance(c, (int, float)) for c in color
        ):
            return ManimColorList(color, alpha)
        else:
            return ManimColor(color, alpha)

    @staticmethod
    def gradient(
        colors: list[ManimColor], length: int
    ) -> ManimColor | list[ManimColor]:
        """This method is currently not implemented. Refer to :func:`color_gradient` for
        a working implementation for now.
        """
        # TODO: implement proper gradient, research good implementation for this or look at 3b1b implementation
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self}')"

    def __str__(self) -> str:
        return f"{self.to_hex()}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            other = ManimColor.gray(other)
        else:
            other = ManimColor(other)
        return np.allclose(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return True

    def __add__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            other = ManimColor.gray(other)
        else:
            other = ManimColor(other)
        alpha = self[3]
        result = np.add(self, other)
        result[3] = alpha
        return result

    def __radd__(self, other: int | float | Self) -> Self:
        return self + other

    def __sub__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            other = ManimColor.gray(other)
        else:
            other = ManimColor(other)
        alpha = self[3]
        result = np.subtract(self, other)
        result[3] = alpha
        return result

    def __rsub__(self, other: int | float | Self) -> Self:
        # 【问题】1 - ManimColor([1,1,1,0.7]) 要等于什么？
        # 【答】1 视为[1/255]，得到ManimColor([(1/255)-1,-1,-1,0.7])保留不透明度
        # 越界截断，所以返回 ManimColor([0,0,0,0.7])
        if isinstance(other, (int, float)):
            other = ManimColor.gray(other)
        else:
            other = ManimColor(other)
        alpha = self[3]
        result = np.subtract(other, self)
        result[3] = alpha
        return result

    def __invert__(self) -> Self:
        return self.invert()

    def __int__(self) -> int:
        return self.to_integer()

    def __hash__(self) -> int:
        return hash(self.to_hex(with_alpha=True))


class ManimColorList(np.ndarray):
    """Represents a list of ManimColors as numpy array.
    输入: colors = [1,2,3]视为一个RGB值
     输出：[[1/255,2/255,3/255,1]]
    输入：colors = [[1,2,3],[4,5,6]]视为两个RGB值
     输出：[[1/255,2/255,3/255,1],[4/255,5/255,6/255,1]]
    输入：colors = [1,2,3,4,5]视为RGBa，
     输出：报错，ManimColor初始化发现color长度不为3或4
    输入多个color和一个opacity，则opacity是所有color的opacity
    若opacity==None且colors不为RGBa，则默认opacity=1（在ManimColor初始化里）
    若opacity不为None 且 colors为RGBa，则opacity覆盖RGBa的a
    """

    def __new__(
        cls,
        colors: (
            np.ndarray
            | Sequence[ManimColor]
            | Sequence[int]
            | Sequence[float]
            | Sequence[str]
            | ManimColor
            | int
            | float
            | str
            | None
        ) = None,
        opacity: float | Sequence[float] | None = None,
    ):
        if isinstance(colors, ManimColorList):
            return colors.copy()
        if colors is None or (
            isinstance(colors, (np.ndarray, Sequence)) and len(colors) == 0
        ):
            return np.array([], dtype=ManimColorDType).view(cls).copy()
        if isinstance(colors, ManimColor):
            obj = np.expand_dims(colors, axis=0).view(cls)
            return obj.copy()
        if isinstance(colors, (int, float, str)):
            # 输入1变为[1]，输入'#'变为['#']
            colors = [colors]
        if not isinstance(colors, (np.ndarray, Sequence)):  # 注意str也是Sequence
            raise ValueError(
                "'colors' must be a list of ManimColor or a single ManimColor",
                f"Your input : colors={colors}",
            )

        if isinstance(colors, np.ndarray):
            # 输入[1,2,3]变为[[1,2,3]]
            if len(colors.shape) == 1:
                colors = [colors]
        elif isinstance(colors, Sequence):
            # 输入[1,2,3]变为[[1,2,3]]，输入[1,2,'#']保持不变
            if all(isinstance(c, (int, float)) for c in colors):
                colors = [colors]

        if not isinstance(opacity, (np.ndarray, Sequence)):
            # 输入多个color和一个opacity，则opacity是所有color的opacity
            color_list = [ManimColor(c, opacity).to_rgba() for c in colors]
        else:
            if len(opacity) != len(colors):
                raise ValueError("length of opacity must match length of color")
            color_list = [ManimColor(c, o).to_rgba() for c, o in zip(colors, opacity)]

        obj = np.asarray(color_list, dtype=np.float64).view(cls)  # ManimColorDType
        return obj.copy()

    # def __array_finalize__(self, obj):
    #     pass

    @property
    def opacity(self) -> np.ndarray:
        # 获取所有颜色的不透明度
        return self[:, 3]

    def append(self, color: ParsableManimColor) -> None:
        """Append a ManimColor to the end of the list."""
        color = ManimColorList(color)
        self.resize((self.shape[0] + color.shape[0], 4), refcheck=False)
        self[-color.shape[0] :, :] = color

    def to_hex_strings(self) -> list[str]:
        return [ManimColor(c).to_hex() for c in self]

    def __eq__(self, other):
        other = ManimColorList(other)
        return np.all(np.array_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        s = "".join(self.to_hex_strings())
        return hash(s)


RGBA = ManimColor
"""RGBA Color Space"""


class HSV(ManimColor):
    """HSV Color Space"""

    def __init__(
        self,
        hsv: HSV_Array_Float | HSV_Tuple_Float | HSVA_Array_Float | HSVA_Tuple_Float,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(None)
        self.__hsv: HSVA_Array_Float
        if len(hsv) == 3:
            self.__hsv = np.asarray((*hsv, alpha))
        elif len(hsv) == 4:
            self.__hsv = np.asarray(hsv)
        else:
            raise ValueError("HSV Color must be an array of 3 values")

    @classmethod
    @override
    def _from_internal(cls, value: ManimColorInternal) -> Self:
        hsv = colorsys.rgb_to_hsv(*value[:3])
        hsva = [*hsv, value[-1]]
        return cls(np.array(hsva))

    @property
    def hue(self) -> float:
        hue: float = self.__hsv[0]
        return hue

    @hue.setter
    def hue(self, hue: float) -> None:
        self.__hsv[0] = hue

    @property
    def saturation(self) -> float:
        saturation: float = self.__hsv[1]
        return saturation

    @saturation.setter
    def saturation(self, saturation: float) -> None:
        self.__hsv[1] = saturation

    @property
    def value(self) -> float:
        value: float = self.__hsv[2]
        return value

    @value.setter
    def value(self, value: float) -> None:
        self.__hsv[2] = value

    @property
    def h(self) -> float:
        hue: float = self.__hsv[0]
        return hue

    @h.setter
    def h(self, hue: float) -> None:
        self.__hsv[0] = hue

    @property
    def s(self) -> float:
        saturation: float = self.__hsv[1]
        return saturation

    @s.setter
    def s(self, saturation: float) -> None:
        self.__hsv[1] = saturation

    @property
    def v(self) -> float:
        value: float = self.__hsv[2]
        return value

    @v.setter
    def v(self, value: float) -> None:
        self.__hsv[2] = value

    @property
    def _internal_space(self) -> npt.NDArray:
        return self.__hsv

    @property
    def _internal_value(self) -> ManimColorInternal:
        """Return the internal value of the current :class:`ManimColor` as an
        ``[r,g,b,a]`` float array.

        Returns
        -------
        ManimColorInternal
            Internal color representation.
        """
        return np.array(
            [
                *colorsys.hsv_to_rgb(self.__hsv[0], self.__hsv[1], self.__hsv[2]),
                self.__alpha,
            ],
            dtype=ManimColorDType,
        )

    @_internal_value.setter
    def _internal_value(self, value: ManimColorInternal) -> None:
        """Overwrite the internal color value of this :class:`ManimColor`.

        Parameters
        ----------
        value
            The value which will overwrite the current color.

        Raises
        ------
        TypeError
            If an invalid array is passed.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Value must be a NumPy array.")
        if value.shape[0] != 4:
            raise TypeError("Array must have exactly 4 values.")
        tmp = colorsys.rgb_to_hsv(value[0], value[1], value[2])
        self.__hsv = np.array(tmp)
        self.__alpha = value[3]


ParsableManimColor: TypeAlias = Union[
    ManimColor,
    int,
    str,
    RGB_Tuple_Int,
    RGB_Tuple_Float,
    RGBA_Tuple_Int,
    RGBA_Tuple_Float,
    RGB_Array_Int,
    RGB_Array_Float,
    RGBA_Array_Int,
    RGBA_Array_Float,
]
"""`ParsableManimColor` represents all the types which can be parsed
to a :class:`ManimColor` in Manim.
"""


ManimColorT = TypeVar("ManimColorT", bound=ManimColor)


def color_to_rgb(color: ParsableManimColor) -> RGB_Array_Float:
    """Transform any form of 'color' to an RGB float array.(RGB from 0.0 to 1.0)

    Parameters
    ----------
    color
        A color to convert to an RGB float array.

    Returns
    -------
    RGB_Array_Float
        The corresponding RGB float array.
    """
    return ManimColor(color).to_rgb()


def color_to_rgba(color: ParsableManimColor, alpha: float = 1.0) -> RGBA_Array_Float:
    """Transform any form of 'color' to an RGBA float array.(RGB from 0.0 to 1.0)

    Parameters
    ----------
    color
        A color to convert to an RGBA float array.
    alpha
        An alpha value between 0.0 and 1.0 to be used as opacity in the color. Default is
        1.0.

    Returns
    -------
    RGBA_Array_Float
        The corresponding RGBA float array.
    """
    return ManimColor(color, alpha).to_rgba()


def color_to_int_rgb(color: ParsableManimColor) -> RGB_Array_Int:
    """Transform any form of 'color' to an RGB integer array.(RGB from 0 to 255)

    Parameters
    ----------
    color
        A color to convert to an RGB integer array.

    Returns
    -------
    RGB_Array_Int
        The corresponding RGB integer array.
    """
    return ManimColor(color).to_int_rgb()


def color_to_int_rgba(color: ParsableManimColor, alpha: float = 1.0) -> RGBA_Array_Int:
    """Transform any form of 'color' to an RGBA integer array.(RGB from 0 to 255)

    Parameters
    ----------
    color
        A color to convert to an RGBA integer array.
    alpha
        An alpha value between 0.0 and 1.0 to be used as opacity in the color. Default is
        1.0.

    Returns
    -------
    RGBA_Array_Int
        The corresponding RGBA integer array.
    """
    return ManimColor(color, alpha).to_int_rgba()


def rgb_to_color(
    rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
) -> ManimColor:
    """Helper function for use in functional style programming. Refer to
    :meth:`ManimColor.from_rgb`.

    Parameters
    ----------
    rgb
        A 3 element iterable.

    Returns
    -------
    ManimColor
        A ManimColor with the corresponding value.
    """
    return ManimColor.from_rgb(rgb)


def rgba_to_color(
    rgba: RGBA_Array_Float | RGBA_Tuple_Float | RGBA_Array_Int | RGBA_Tuple_Int,
) -> ManimColor:
    """Helper function for use in functional style programming. Refer to
    :meth:`ManimColor.from_rgba`.

    Parameters
    ----------
    rgba
        A 4 element iterable.

    Returns
    -------
    ManimColor
        A ManimColor with the corresponding value
    """
    return ManimColor.from_rgba(rgba)


def rgb_to_hex(
    rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
) -> str:
    """Helper function for use in functional style programming. Refer to
    :meth:`ManimColor.from_rgb` and :meth:`ManimColor.to_hex`.

    Parameters
    ----------
    rgb
        A 3 element iterable.

    Returns
    -------
    str
        A hex representation of the color.
    """
    return ManimColor.from_rgb(rgb).to_hex()


def hex_to_rgb(hex_code: str) -> RGB_Array_Float:
    """Helper function for use in functional style programming. Refer to
    :meth:`ManimColor.to_rgb`.

    Parameters
    ----------
    hex_code
        A hex string representing a color.

    Returns
    -------
    RGB_Array_Float
        An RGB array representing the color.
    """
    return ManimColor(hex_code).to_rgb()


def invert_color(color: ManimColorT) -> ManimColorT:
    """Helper function for use in functional style programming. Refer to
    :meth:`ManimColor.invert`

    Parameters
    ----------
    color
        The :class:`ManimColor` to invert.

    Returns
    -------
    ManimColor
        The linearly inverted :class:`ManimColor`.
    """
    return color.invert()


def color_gradient(
    reference_colors: Sequence[ParsableManimColor],
    length_of_output: int,
) -> list[ManimColor] | ManimColor:
    """Create a list of colors interpolated between the input array of colors with a
    specific number of colors.

    Parameters
    ----------
    reference_colors
        The colors to be interpolated between or spread apart.
    length_of_output
        The number of colors that the output should have, ideally more than the input.

    Returns
    -------
    list[ManimColor] | ManimColor
        A :class:`ManimColor` or a list of interpolated :class:`ManimColor`'s.
    """
    if length_of_output == 0:
        return ManimColor(reference_colors[0])
    if len(reference_colors) == 1:
        return [ManimColor(reference_colors[0])] * length_of_output
    rgbs = [color_to_rgb(color) for color in reference_colors]
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype("int")
    alphas_mod1 = alphas % 1
    # End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color((rgbs[i] * (1 - alpha)) + (rgbs[i + 1] * alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]


def interpolate_color(
    color1: ManimColorT, color2: ManimColorT, alpha: float
) -> ManimColorT:
    """Standalone function to interpolate two ManimColors and get the result. Refer to
    :meth:`ManimColor.interpolate`.

    Parameters
    ----------
    color1
        The first :class:`ManimColor`.
    color2
        The second :class:`ManimColor`.
    alpha
        The alpha value determining the point of interpolation between the colors.

    Returns
    -------
    ManimColor
        The interpolated ManimColor.
    """
    return color1.interpolate(color2, alpha)


def average_color(*colors: ParsableManimColor) -> ManimColor:
    """Determine the average color between the given parameters.

    .. note::
        This operation does not consider the alphas (opacities) of the colors. The
        generated color has an alpha or opacity of 1.0.

    Returns
    -------
    ManimColor
        The average color of the input.
    """
    rgbs = np.array([color_to_rgb(color) for color in colors])
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> ManimColor:
    """Return a random bright color: a random color averaged with ``WHITE``.

    .. warning::
        This operation is very expensive. Please keep in mind the performance loss.

    Returns
    -------
    ManimColor
        A random bright :class:`ManimColor`.
    """
    curr_rgb = color_to_rgb(random_color())
    new_rgb = 0.5 * (curr_rgb + np.ones(3))
    return ManimColor(new_rgb)


def random_color() -> ManimColor:
    """Return a random :class:`ManimColor`.

    .. warning::
        This operation is very expensive. Please keep in mind the performance loss.

    Returns
    -------
    ManimColor
        A random :class:`ManimColor`.
    """
    import manim.utils.color.manim_colors as manim_colors

    return random.choice(manim_colors._all_manim_colors)


def get_shaded_rgb(
    rgb: RGB_Array_Float,
    point: Point3D,
    unit_normal_vect: Vector3D,
    light_source: Point3D,
) -> RGB_Array_Float:
    """Add light or shadow to the ``rgb`` color of some surface which is located at a
    given ``point`` in space and facing in the direction of ``unit_normal_vect``,
    depending on whether the surface is facing a ``light_source`` or away from it.

    Parameters
    ----------
    rgb
        An RGB array of floats.
    point
        The location of the colored surface.
    unit_normal_vect
        The direction in which the colored surface is facing.
    light_source
        The location of a light source which might illuminate the surface.

    Returns
    -------
    RGB_Array_Float
        The color with added light or shadow, depending on the direction of the colored
        surface.
    """
    to_sun = normalize(light_source - point)
    light = 0.5 * np.dot(unit_normal_vect, to_sun) ** 3
    if light < 0:
        light *= 0.5
    shaded_rgb: RGB_Array_Float = rgb + light
    return shaded_rgb


__all__ = [
    "ManimColor",
    "ManimColorList",
    "ManimColorDType",
    "ParsableManimColor",
    "color_to_rgb",
    "color_to_rgba",
    "color_to_int_rgb",
    "color_to_int_rgba",
    "rgb_to_color",
    "rgba_to_color",
    "rgb_to_hex",
    "hex_to_rgb",
    "invert_color",
    "color_gradient",
    "interpolate_color",
    "average_color",
    "random_bright_color",
    "random_color",
    "get_shaded_rgb",
    "HSV",
    "RGBA",
]
