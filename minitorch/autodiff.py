import uuid
import queue

def wrap_tuple(x):
	if isinstance(x, tuple):
		return x
	return (x,)


def unwrap_tuple(x):
	if len(x) == 1:
		return x[0]
	return x


class Variable:
	"""
	Attributes:
		history (:class:`History`) : the Function calls that created this variable or None if constant
		derivative (number): the derivative with respect to this variable
		name (string) : an optional name for debugging
	"""

	def __init__(self, history, name=None):
		assert history is None or isinstance(history, History), history

		self.history = history
		self._derivative = None

		# For debugging can have a name.
		if name is not None:
			self.name = name
		else:
			self.name = str(uuid.uuid4())

	def requires_grad_(self, val):
		self.history = History(None, None, None)

	def backward(self, d_output=None):
		"""
		Calls autodiff to fill in the derivatives for the history of this object.
		"""
		if d_output is None:
			d_output = 1.0
		backpropagate(VariableWithDeriv(self, d_output))

	@property
	def derivative(self):
		return self._derivative

	## IGNORE
	def __hash__(self):
		return hash(self._name)

	def _add_deriv(self, val):
		assert self.history.is_leaf(), "Only leaf variables can have derivatives."
		if self._derivative is None:
			self._derivative = self.zeros()
		self._derivative += val

	def zero_grad_(self):
		self._derivative = self.zeros()

	def __radd__(self, b):
		return self + b

	def __rmul__(self, b):
		return self * b

	def zeros(self):
		return 0.0

	def expand(self, x):
		return x

	## IGNORE


class Context:
	"""
	Context class is used by.
	"""

	def __init__(self, no_grad=False):
		self._saved_values = None
		self.no_grad = no_grad

	def save_for_backward(self, *values):
		if self.no_grad:
			return
		self._saved_values = values

	@property
	def saved_values(self):
		assert not self.no_grad, "Doesn't require grad"
		assert self._saved_values is not None, "Did you forget to save values?"
		return unwrap_tuple(self._saved_values)


class History:
	"""
	`History` stores all of the `Function` operations that were used to
	construct an autodiff object.

	Attributes:
		last_fn (:class:`FunctionBase`) : The last function that was called.
		ctx (:class:`Context`): The context for that function.
		inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.
	"""

	def __init__(self, last_fn=None, ctx=None, inputs=None):
		self.last_fn = last_fn
		self.ctx = ctx
		self.inputs = inputs

	def is_leaf(self):
		return self.last_fn is None

	def backprop_step(self, d_output):
		return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


class VariableWithDeriv:
	"Holder for a variable with it derivative."

	def __init__(self, variable, deriv):
		self.variable = variable
		self.deriv = variable.expand(deriv)


class FunctionBase:
	"""
	A function that can act on :class:`Variable` arguments to
	produce a :class:`Variable` output, while tracking the internal history.

	Call by :func:`FunctionBase.apply`.

	"""

	@staticmethod
	def variable(raw, history):
		pass

	@classmethod
	def apply(cls, *vals):
		raw_vals = []
		need_grad = False
		for v in vals:
			if isinstance(v, Variable):
				if v.history is not None:
					need_grad = True
				raw_vals.append(v.get_data())
			else:
				raw_vals.append(v)
		ctx = Context(not need_grad)
		c = cls.forward(ctx, *raw_vals)
		assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
			cls.data_type,
			type(c),
		)
		back = None
		if need_grad:
			back = History(cls, ctx, vals)
		return cls.variable(cls.data(c), back)

	@classmethod
	def chain_rule(cls, ctx, inputs, d_output):
		# TODO: Implement for Task 1.3.
		# ~ raise NotImplementedError('Need to implement for Task 1.3')
		par = [] # arreglo de VariableWithDeriv (Variable + derivada)
		derivadas = cls.backward(ctx, d_output)
		if not isinstance(derivadas, tuple):
			derivadas = wrap_tuple(derivadas)
		"""
		Se aplica la regla de la cadena
		"""
		for i, variable in enumerate(inputs):
			if not is_constant(variable):
				par.append(VariableWithDeriv(variable, derivadas[i]))
		return par

def is_leaf(val):
	return isinstance(val, Variable) and val.history.is_leaf()


def is_constant(val):
	return not isinstance(val, Variable) or val.history is None


def backpropagate(final_variable_with_deriv):
	"""
	Runs a breadth-first search on the computation graph in order to
	backpropagate derivatives to the leaves.

	See :doc:`backpropagate` for details on the algorithm

	Args:
	   final_variable_with_deriv (:class:`VariableWithDeriv`): The final variable
		   and its derivative that we want to propagate backward to the leaves.
	"""
	# TODO: Implement for Task 1.4.
	# ~ raise NotImplementedError('Need to implement for Task 1.4')
	""" 
	Hay que seguir paso a paso el algoritmo que se encuentra en la documentación de Minitorch: 
	0. Inicializar una cola con el par final Variable + derivada
	1. Mientras la cola no está vacía, extraiga una variable + derivada de la cola:
		a. si la Variable es una hoja, agregue su derivada final ( _add_deriv ) y haga un bucle en (1)
		b. si la Variable no es una hoja,
			1. llamar .chain_rule en la última función que lo creó con derivativo comodout
			2. recorrer todas las Variables + derivada producida por la regla de la cadena (eliminando constantes)
			3. opcional, si la Variable está en la cola (verifique .name ), agregue a su derivada actual;
			4. de lo contrario, agregue a la cola.
	"""
	#Paso 0
	cola = queue.Queue()
	cola.put(final_variable_with_deriv)
	#Paso 1
	while not(cola.empty()):
		par = cola.get() # el "par" consta de una variable y la derivada respecto de esa variable
		#Paso 1.a
		if is_leaf(par.variable):
			par.variable._add_deriv(par.deriv)
		#Paso 1.b
		else:
			historico = par.variable.history
			assert historico is not None
			paresHistoricos = historico.backprop_step(par.deriv)
			for parIterador in paresHistoricos:
				if not is_constant(parIterador.variable):
					cola.put(parIterador)
