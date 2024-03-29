# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
mrcinterpreter
--------------

Module which exports the :class:`MrcInterpreter` class.

Classes:
    :class:`MrcInterpreter`: An object which can interpret an I/O stream as MRC
    data.

"""

# Import Python 3 features for future-proofing
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings

import numpy as np

from . import utils
from .dtypes import HEADER_DTYPE, FEI_EXTENDED_HEADER_DTYPE
from .mrcobject import MrcObject
from .constants import MAP_ID


class MrcInterpreter(MrcObject):
    
    """An object which interprets an I/O stream as MRC / CCP4 map data.
    
    The header and data are handled as numpy arrays - see
    :class:`~mrcfile.mrcobject.MrcObject` for details.
    
    :class:`MrcInterpreter` can be used directly, but it is mostly intended as
    a superclass to provide common stream-handling functionality. This can be
    used by subclasses which will handle opening and closing the stream.
    
    This class implements the :meth:`__enter__` and :meth:`__exit__` special
    methods which allow it to be used by the Python context manager in a
    :keyword:`with` block. This ensures that :meth:`close` is called after the
    object is finished with.
        
    When reading the I/O stream, a :class:`~exceptions.ValueError` is raised if
    the data is invalid in one of the following ways:
    
    #. The header's ``map`` field is not set correctly to confirm the file
       type.
    #. The machine stamp is invalid and so the data's byte order cannot be
       determined.
    #. The mode number is not recognised. Currently accepted modes are 0, 1, 2,
       4 and 6.
    #. The data block is not large enough for the specified data type and
       dimensions.
    
    :class:`MrcInterpreter` offers a permissive read mode for handling
    problematic files. If ``permissive`` is set to :data:`True` and any of the
    validity checks fails, a :mod:`warning <warnings>` is issued instead of an
    exception, and file interpretation continues. If the mode number is invalid
    or the data block is too small, the :attr:`data` attribute will be set to
    :data:`None`. In this case, it might be possible to inspect and correct the
    header, and then call :meth:`_read` again to read the data correctly. See
    the :doc:`usage guide <../usage_guide>` for more details.
    
    Methods:
    
    * :meth:`flush`
    * :meth:`close`
    
    Methods relevant to subclasses:
    
    * :meth:`_read`
    * :meth:`_read_data`
    
    """
    
    def __init__(self, iostream=None, permissive=False, **kwargs):
        """Initialise a new MrcInterpreter object.
        
        This initialiser reads the stream if it is given. In general,
        subclasses should call :meth:`super().__init__` without giving an
        ``iostream`` argument, then set the :attr:`_iostream` attribute
        themselves and call :meth:`_read` when ready.
        
        To use the MrcInterpreter class directly, pass a stream when creating
        the object (or for a write-only stream, create an MrcInterpreter with
        no stream, call :meth:`_create_default_attributes` and set the
        :attr:`_iostream` attribute directly).
        
        Args:
            iostream: The I/O stream to use to read and write MRC data. The
                default is :data:`None`.
            permissive: Read the stream in permissive mode. The default is
                :data:`False`.
        
        Raises:
            :class:`~exceptions.ValueError`: If ``iostream`` is given and the
                data it contains cannot be interpreted as a valid MRC file.
        """
        super(MrcInterpreter, self).__init__(**kwargs)
        
        self._iostream = iostream
        self._permissive = permissive
        
        # If iostream is given, initialise by reading it
        if self._iostream is not None:
            self._read()
    
    def __enter__(self):
        """Called by the context manager at the start of a :keyword:`with`
        block.
        
        Returns:
            This object (``self``).
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called by the context manager at the end of a :keyword:`with`
        block.
        
        This ensures that the :meth:`close` method is called.
        """
        self.close()
    
    def __del__(self):
        """Attempt to flush the stream when this object is garbage collected.
        
        It's better not to rely on this - instead, use a :keyword:`with`
        block or explicitly call the :meth:`close` method.
        """
        try:
            self.close()
        except Exception:
            pass
    
    def _read(self):
        """Read the header, extended header and data from the I/O stream.
        
        Before calling this method, the stream should be open and positioned at
        the start of the header. This method will advance the stream to the end
        of the data block.
        
        Raises:
            :class:`~exceptions.ValueError`: If the file is not a valid MRC
                file.
        """
        self._read_header()
        self._read_extended_header()
        self._read_data()

    def _read_header(self):
        """Read the MRC header from the I/O stream.
        
        The header will be read from the current stream position, and the
        stream will be advanced by 1024 bytes.
        
        Raises:
            :class:`~exceptions.ValueError`: If the file is not a valid MRC
                file.
        """
        # Read 1024 bytes from the stream
        header_str = self._iostream.read(HEADER_DTYPE.itemsize)
        
        if len(header_str) < HEADER_DTYPE.itemsize:
            raise ValueError("Couldn't read enough bytes for MRC header")
        
        # Use a recarray to allow access to fields as attributes
        # (e.g. header.mode instead of header['mode'])
        header = np.rec.fromstring(header_str, dtype=HEADER_DTYPE, shape=())
        
        # Make header writeable, because fromstring() creates a read-only array
        header.flags.writeable = True
        
        # Check this is an MRC file, and read machine stamp to get byte order
        if header.map != MAP_ID:
            msg = ("Map ID string not found - "
                   "not an MRC file, or file is corrupt")
            if self._permissive:
                warnings.warn(msg, RuntimeWarning)
            else:
                raise ValueError(msg)
        
        try:
            byte_order = utils.byte_order_from_machine_stamp(header.machst)
        except ValueError as err:
            if self._permissive:
                byte_order = '<' # try little-endian as a sensible default
                warnings.warn(str(err), RuntimeWarning)
            else:
                raise
        
        # Create a new dtype with the correct byte order and update the header
        header.dtype = header.dtype.newbyteorder(byte_order)
        
        header.flags.writeable = not self._read_only
        self._header = header
    
    def _read_extended_header(self):
        """Read the extended header from the stream.
        
        If there is no extended header, a zero-length array is assigned to the
        extended_header attribute.
        
        If the extended header is recognised as FEI microscope metadata (by
        'FEI1' in the header's ``exttyp`` field), its dtype is set
        appropriately. Otherwise, the dtype is set as void (``'V1'``).
        """
        ext_header_str = self._iostream.read(int(self.header.nsymbt))
        
        if self.header['exttyp'] == b'FEI1':
            dtype = FEI_EXTENDED_HEADER_DTYPE
        else:
            dtype = 'V1'
            
        self._extended_header = np.frombuffer(ext_header_str, dtype=dtype)
        self._extended_header.flags.writeable = not self._read_only
    
    def _read_data(self):
        """Read the data array from the stream.
        
        This method uses information from the header to set the data array's
        shape and dtype.
        """
        try:
            dtype = utils.data_dtype_from_header(self.header)
        except ValueError as err:
            if self._permissive:
                warnings.warn("{0} - data block cannot be read".format(err),
                              RuntimeWarning)
                self._data = None
                return
            else:
                raise
        
        shape = utils.data_shape_from_header(self.header)
        
        nbytes = dtype.itemsize
        for axis_length in shape:
            nbytes *= axis_length
        
        data_bytes = self._iostream.read(nbytes)
        
        if len(data_bytes) < nbytes:
            msg = ("Expected {0} bytes in data block but could only read {1}"
                   .format(nbytes, len(data_bytes)))
            if self._permissive:
                warnings.warn(msg, RuntimeWarning)
                self._data = None
                return
            else:
                raise ValueError(msg)
        
        self._data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        self._data.flags.writeable = not self._read_only
    
    def close(self):
        """Flush to the stream and clear the header and data attributes."""
        if self._header is not None and not self._iostream.closed:
            self.flush()
        self._header = None
        self._extended_header = None
        self._close_data()
    
    def flush(self):
        """Flush the header and data arrays to the I/O stream.
        
        This implementation seeks to the start of the stream, writes the
        header, extended header and data arrays, and then truncates the stream.
        
        Subclasses should override this implementation for streams which do not
        support :meth:`~io.IOBase.seek` or :meth:`~io.IOBase.truncate`.
        """
        if not self._read_only:
            self._iostream.seek(0)
            self._iostream.write(self.header)
            self._iostream.write(self.extended_header)
            self._iostream.write(np.ascontiguousarray(self.data))
            self._iostream.truncate()
            self._iostream.flush()
