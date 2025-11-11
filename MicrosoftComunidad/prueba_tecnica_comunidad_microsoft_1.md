# Prueba Técnica - Comunidad Microsoft

## Objetivo y tiempo
- Tiempo estimado: 40 minutos.
- Objetivo: implementar una solución pequeña que demuestre tu conocimiento y capacidad de resolver problemas, en cualquier lenguaje.

## Alcance limitado (elige HTTP o CLI)
- Opción A (HTTP): crea una API mínima con estos 3 endpoints:
  - `GET /events` → lista de eventos.
  - `POST /events` → crea un evento (validaciones básicas).
  - `POST /events/{id}/register` → inscribe un usuario por email.
- Opción B (CLI): comandos equivalentes:
  - `events list`, `events create`, `events register <id> --email <email>`.
- Persistencia: puede ser en memoria o archivo local (BD opcional).


# Enlace a chat gpt para la creación de API por medio de HTTP
https://chatgpt.com/share/6912b828-cc84-800e-833d-abaa69c804e1

# HTTP
# 1. crear archivo requirements.txt
requirements.txt
fastapi==0.114.2
uvicorn==0.30.6
pydantic==2.9.2

# 2.crear archivo app.py
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
from fastapi import FastAPI, HTTPException, Path, Body, Query
from pydantic import BaseModel, Field, EmailStr, field_validator

app = FastAPI(title="Eventos API", version="1.0.0")


# -----------------------------
# Modelos
# -----------------------------
class ErrorResponse(BaseModel):
    status: str = Field(default="error")
    message: str


class EventCreate(BaseModel):
    titulo: str = Field(min_length=1, max_length=200)
    fecha_inicio: datetime
    fecha_fin: datetime
    cupo: int = Field(ge=0)

    @field_validator("fecha_fin")
    @classmethod
    def validar_fechas(cls, v: datetime, info):
        # fecha_fin >= fecha_inicio
        data = info.data
        fi = data.get("fecha_inicio")
        if fi and v < fi:
            raise ValueError("fecha_fin debe ser mayor o igual a fecha_inicio")
        return v


class EventOut(BaseModel):
    id: int
    titulo: str
    fecha_inicio: datetime
    fecha_fin: datetime
    cupo: int
    inscritos_count: int


class RegisterRequest(BaseModel):
    email: EmailStr


class EventsResponse(BaseModel):
    status: str = "ok"
    data: List[EventOut]
    page: int
    limit: int
    total: int


class OkResponse(BaseModel):
    status: str = "ok"
    message: str
    data: Optional[Dict[str, Any]] = None


# -----------------------------
# "Persistencia" en memoria
# -----------------------------
_events: Dict[int, Dict[str, Any]] = {}
_registrations: Set[Tuple[int, str]] = set()  # (evento_id, email)
_next_id: int = 1


def _to_event_out(e: Dict[str, Any]) -> EventOut:
    return EventOut(
        id=e["id"],
        titulo=e["titulo"],
        fecha_inicio=e["fecha_inicio"],
        fecha_fin=e["fecha_fin"],
        cupo=e["cupo"],
        inscritos_count=e["inscritos_count"],
    )


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/events", response_model=EventsResponse, responses={400: {"model": ErrorResponse}})
def list_events(
    q: Optional[str] = Query(default=None, description="Filtro por texto en título (opcional)"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
):
    """
    Lista de eventos con filtro de texto y paginación simple.
    """
    events_list = [_to_event_out(e) for e in _events.values()]
    if q:
        q_lower = q.lower()
        events_list = [e for e in events_list if q_lower in e.titulo.lower()]

    total = len(events_list)
    start = (page - 1) * limit
    end = start + limit
    paginated = events_list[start:end]

    return EventsResponse(status="ok", data=paginated, page=page, limit=limit, total=total)


@app.post(
    "/events",
    response_model=OkResponse,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def create_event(payload: EventCreate = Body(...)):
    """
    Crea un evento.
    Reglas:
    - fecha_fin >= fecha_inicio (validado por Pydantic)
    - cupo >= 0 (validado por Pydantic)
    """
    global _next_id
    new_event = {
        "id": _next_id,
        "titulo": payload.titulo.strip(),
        "fecha_inicio": payload.fecha_inicio,
        "fecha_fin": payload.fecha_fin,
        "cupo": payload.cupo,
        "inscritos_count": 0,
    }
    _events[_next_id] = new_event
    _next_id += 1

    return OkResponse(
        message="Evento creado",
        data={"event": _to_event_out(new_event).model_dump()},
    )


@app.post(
    "/events/{id}/register",
    response_model=OkResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def register_user(
    id: int = Path(..., ge=1),
    payload: RegisterRequest = Body(...),
):
    """
    Inscribe un usuario (email) a un evento.
    Reglas:
    - Evento debe existir
    - No permitir duplicados (mismo email en el mismo evento)
    - No inscribir si cupo está completo
    """
    event = _events.get(id)
    if not event:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Evento no encontrado"})

    key = (id, payload.email.lower())
    if key in _registrations:
        raise HTTPException(
            status_code=409,
            detail={"status": "error", "message": "El usuario ya está inscrito en este evento"},
        )

    if event["inscritos_count"] >= event["cupo"]:
        raise HTTPException(
            status_code=409,
            detail={"status": "error", "message": "Cupo completo"},
        )

    _registrations.add(key)
    event["inscritos_count"] += 1

    return OkResponse(
        message="Inscripción exitosa",
        data={
            "event": _to_event_out(event).model_dump(),
            "email": payload.email.lower(),
        },
    )


# -----------------------------
# Manejo de errores consistente
# (Transforma detail dict -> ErrorResponse)
# -----------------------------
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import HTTPException as FastAPIHTTPException


@app.exception_handler(FastAPIHTTPException)
async def custom_http_exception_handler(request: Request, exc: FastAPIHTTPException):
    if isinstance(exc.detail, dict):
        # Ya viene con estructura {status, message}
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    # Fallback
    return await http_exception_handler(request, exc)


# 3. Crear venv 
python -m venv .venv

# activar venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

pip install pydantic[email]


# 4. Ejecutar
uvicorn app:app --reload --port 8000

# Dar click en 
 http://127.0.0.1:8000



# 4. networks.py
"""The networks module contains types for common network-related fields."""

from __future__ import annotations as _annotations

import dataclasses as _dataclasses
import re
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Any

from pydantic_core import MultiHostUrl, PydanticCustomError, Url, core_schema
from typing_extensions import Annotated, Self, TypeAlias

from ._internal import _fields, _repr, _schema_generation_shared
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler
from .json_schema import JsonSchemaValue

if TYPE_CHECKING:
    import email_validator

    NetworkType: TypeAlias = 'str | bytes | int | tuple[str | bytes | int, str | int]'

else:
    email_validator = None


__all__ = [
    'AnyUrl',
    'AnyHttpUrl',
    'FileUrl',
    'FtpUrl',
    'HttpUrl',
    'WebsocketUrl',
    'AnyWebsocketUrl',
    'UrlConstraints',
    'EmailStr',
    'NameEmail',
    'IPvAnyAddress',
    'IPvAnyInterface',
    'IPvAnyNetwork',
    'PostgresDsn',
    'CockroachDsn',
    'AmqpDsn',
    'RedisDsn',
    'MongoDsn',
    'KafkaDsn',
    'NatsDsn',
    'validate_email',
    'MySQLDsn',
    'MariaDBDsn',
    'ClickHouseDsn',
    'SnowflakeDsn',
]


@_dataclasses.dataclass
class UrlConstraints(_fields.PydanticMetadata):
    """Url constraints.

    Attributes:
        max_length: The maximum length of the url. Defaults to `None`.
        allowed_schemes: The allowed schemes. Defaults to `None`.
        host_required: Whether the host is required. Defaults to `None`.
        default_host: The default host. Defaults to `None`.
        default_port: The default port. Defaults to `None`.
        default_path: The default path. Defaults to `None`.
    """

    max_length: int | None = None
    allowed_schemes: list[str] | None = None
    host_required: bool | None = None
    default_host: str | None = None
    default_port: int | None = None
    default_path: str | None = None

    def __hash__(self) -> int:
        return hash(
            (
                self.max_length,
                tuple(self.allowed_schemes) if self.allowed_schemes is not None else None,
                self.host_required,
                self.default_host,
                self.default_port,
                self.default_path,
            )
        )


AnyUrl = Url
"""Base type for all URLs.

* Any scheme allowed
* Top-level domain (TLD) not required
* Host required

Assuming an input URL of `http://samuel:pass@example.com:8000/the/path/?query=here#fragment=is;this=bit`,
the types export the following properties:

- `scheme`: the URL scheme (`http`), always set.
- `host`: the URL host (`example.com`), always set.
- `username`: optional username if included (`samuel`).
- `password`: optional password if included (`pass`).
- `port`: optional port (`8000`).
- `path`: optional path (`/the/path/`).
- `query`: optional URL query (for example, `GET` arguments or "search string", such as `query=here`).
- `fragment`: optional fragment (`fragment=is;this=bit`).
"""
AnyHttpUrl = Annotated[Url, UrlConstraints(allowed_schemes=['http', 'https'])]
"""A type that will accept any http or https URL.

* TLD not required
* Host required
"""
HttpUrl = Annotated[Url, UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])]
"""A type that will accept any http or https URL.

* TLD not required
* Host required
* Max length 2083

```py
from pydantic import BaseModel, HttpUrl, ValidationError

class MyModel(BaseModel):
    url: HttpUrl

m = MyModel(url='http://www.example.com')  # (1)!
print(m.url)
#> http://www.example.com/

try:
    MyModel(url='ftp://invalid.url')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyModel
    url
      URL scheme should be 'http' or 'https' [type=url_scheme, input_value='ftp://invalid.url', input_type=str]
    '''

try:
    MyModel(url='not a url')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyModel
    url
      Input should be a valid URL, relative URL without a base [type=url_parsing, input_value='not a url', input_type=str]
    '''
```

1. Note: mypy would prefer `m = MyModel(url=HttpUrl('http://www.example.com'))`, but Pydantic will convert the string to an HttpUrl instance anyway.

"International domains" (e.g. a URL where the host or TLD includes non-ascii characters) will be encoded via
[punycode](https://en.wikipedia.org/wiki/Punycode) (see
[this article](https://www.xudongz.com/blog/2017/idn-phishing/) for a good description of why this is important):

```py
from pydantic import BaseModel, HttpUrl

class MyModel(BaseModel):
    url: HttpUrl

m1 = MyModel(url='http://puny£code.com')
print(m1.url)
#> http://xn--punycode-eja.com/
m2 = MyModel(url='https://www.аррӏе.com/')
print(m2.url)
#> https://www.xn--80ak6aa92e.com/
m3 = MyModel(url='https://www.example.珠宝/')
print(m3.url)
#> https://www.example.xn--pbt977c/
```


!!! warning "Underscores in Hostnames"
    In Pydantic, underscores are allowed in all parts of a domain except the TLD.
    Technically this might be wrong - in theory the hostname cannot have underscores, but subdomains can.

    To explain this; consider the following two cases:

    - `exam_ple.co.uk`: the hostname is `exam_ple`, which should not be allowed since it contains an underscore.
    - `foo_bar.example.com` the hostname is `example`, which should be allowed since the underscore is in the subdomain.

    Without having an exhaustive list of TLDs, it would be impossible to differentiate between these two. Therefore
    underscores are allowed, but you can always do further validation in a validator if desired.

    Also, Chrome, Firefox, and Safari all currently accept `http://exam_ple.com` as a URL, so we're in good
    (or at least big) company.
"""
AnyWebsocketUrl = Annotated[Url, UrlConstraints(allowed_schemes=['ws', 'wss'])]
"""A type that will accept any ws or wss URL.

* TLD not required
* Host required
"""
WebsocketUrl = Annotated[Url, UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])]
"""A type that will accept any ws or wss URL.

* TLD not required
* Host required
* Max length 2083
"""
FileUrl = Annotated[Url, UrlConstraints(allowed_schemes=['file'])]
"""A type that will accept any file URL.

* Host not required
"""
FtpUrl = Annotated[Url, UrlConstraints(allowed_schemes=['ftp'])]
"""A type that will accept ftp URL.

* TLD not required
* Host required
"""
PostgresDsn = Annotated[
    MultiHostUrl,
    UrlConstraints(
        host_required=True,
        allowed_schemes=[
            'postgres',
            'postgresql',
            'postgresql+asyncpg',
            'postgresql+pg8000',
            'postgresql+psycopg',
            'postgresql+psycopg2',
            'postgresql+psycopg2cffi',
            'postgresql+py-postgresql',
            'postgresql+pygresql',
        ],
    ),
]
"""A type that will accept any Postgres DSN.

* User info required
* TLD not required
* Host required
* Supports multiple hosts

If further validation is required, these properties can be used by validators to enforce specific behaviour:

```py
from pydantic import (
    BaseModel,
    HttpUrl,
    PostgresDsn,
    ValidationError,
    field_validator,
)

class MyModel(BaseModel):
    url: HttpUrl

m = MyModel(url='http://www.example.com')

# the repr() method for a url will display all properties of the url
print(repr(m.url))
#> Url('http://www.example.com/')
print(m.url.scheme)
#> http
print(m.url.host)
#> www.example.com
print(m.url.port)
#> 80

class MyDatabaseModel(BaseModel):
    db: PostgresDsn

    @field_validator('db')
    def check_db_name(cls, v):
        assert v.path and len(v.path) > 1, 'database must be provided'
        return v

m = MyDatabaseModel(db='postgres://user:pass@localhost:5432/foobar')
print(m.db)
#> postgres://user:pass@localhost:5432/foobar

try:
    MyDatabaseModel(db='postgres://user:pass@localhost:5432')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyDatabaseModel
    db
      Assertion failed, database must be provided
    assert (None)
     +  where None = MultiHostUrl('postgres://user:pass@localhost:5432').path [type=assertion_error, input_value='postgres://user:pass@localhost:5432', input_type=str]
    '''
```
"""

CockroachDsn = Annotated[
    Url,
    UrlConstraints(
        host_required=True,
        allowed_schemes=[
            'cockroachdb',
            'cockroachdb+psycopg2',
            'cockroachdb+asyncpg',
        ],
    ),
]
"""A type that will accept any Cockroach DSN.

* User info required
* TLD not required
* Host required
"""
AmqpDsn = Annotated[Url, UrlConstraints(allowed_schemes=['amqp', 'amqps'])]
"""A type that will accept any AMQP DSN.

* User info required
* TLD not required
* Host required
"""
RedisDsn = Annotated[
    Url,
    UrlConstraints(allowed_schemes=['redis', 'rediss'], default_host='localhost', default_port=6379, default_path='/0'),
]
"""A type that will accept any Redis DSN.

* User info required
* TLD not required
* Host required (e.g., `rediss://:pass@localhost`)
"""
MongoDsn = Annotated[MultiHostUrl, UrlConstraints(allowed_schemes=['mongodb', 'mongodb+srv'], default_port=27017)]
"""A type that will accept any MongoDB DSN.

* User info not required
* Database name not required
* Port not required
* User info may be passed without user part (e.g., `mongodb://mongodb0.example.com:27017`).
"""
KafkaDsn = Annotated[Url, UrlConstraints(allowed_schemes=['kafka'], default_host='localhost', default_port=9092)]
"""A type that will accept any Kafka DSN.

* User info required
* TLD not required
* Host required
"""
NatsDsn = Annotated[
    MultiHostUrl,
    UrlConstraints(allowed_schemes=['nats', 'tls', 'ws', 'wss'], default_host='localhost', default_port=4222),
]
"""A type that will accept any NATS DSN.

NATS is a connective technology built for the ever increasingly hyper-connected world.
It is a single technology that enables applications to securely communicate across
any combination of cloud vendors, on-premise, edge, web and mobile, and devices.
More: https://nats.io
"""
MySQLDsn = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=[
            'mysql',
            'mysql+mysqlconnector',
            'mysql+aiomysql',
            'mysql+asyncmy',
            'mysql+mysqldb',
            'mysql+pymysql',
            'mysql+cymysql',
            'mysql+pyodbc',
        ],
        default_port=3306,
    ),
]
"""A type that will accept any MySQL DSN.

* User info required
* TLD not required
* Host required
"""
MariaDBDsn = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=['mariadb', 'mariadb+mariadbconnector', 'mariadb+pymysql'],
        default_port=3306,
    ),
]
"""A type that will accept any MariaDB DSN.

* User info required
* TLD not required
* Host required
"""
ClickHouseDsn = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=['clickhouse+native', 'clickhouse+asynch'],
        default_host='localhost',
        default_port=9000,
    ),
]
"""A type that will accept any ClickHouse DSN.

* User info required
* TLD not required
* Host required
"""
SnowflakeDsn = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=['snowflake'],
        host_required=True,
    ),
]
"""A type that will accept any Snowflake DSN.

* User info required
* TLD not required
* Host required
"""


def import_email_validator() -> None:
    global email_validator
    try:
        import email_validator
    except ImportError as e:
        raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e
    if not version('email-validator').partition('.')[0] == '2':
        raise ImportError('email-validator version >= 2.0 required, run pip install -U email-validator')


if TYPE_CHECKING:
    EmailStr = Annotated[str, ...]
else:

    class EmailStr:
        """
        Info:
            To use this type, you need to install the optional
            [`email-validator`](https://github.com/JoshData/python-email-validator) package:

            ```bash
            pip install email-validator
            ```

        Validate email addresses.

        ```py
        from pydantic import BaseModel, EmailStr

        class Model(BaseModel):
            email: EmailStr

        print(Model(email='contact@mail.com'))
        #> email='contact@mail.com'
        ```
        """  # noqa: D212

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source: type[Any],
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            import_email_validator()
            return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            field_schema = handler(core_schema)
            field_schema.update(type='string', format='email')
            return field_schema

        @classmethod
        def _validate(cls, input_value: str, /) -> str:
            return validate_email(input_value)[1]


class NameEmail(_repr.Representation):
    """
    Info:
        To use this type, you need to install the optional
        [`email-validator`](https://github.com/JoshData/python-email-validator) package:

        ```bash
        pip install email-validator
        ```

    Validate a name and email address combination, as specified by
    [RFC 5322](https://datatracker.ietf.org/doc/html/rfc5322#section-3.4).

    The `NameEmail` has two properties: `name` and `email`.
    In case the `name` is not provided, it's inferred from the email address.

    ```py
    from pydantic import BaseModel, NameEmail

    class User(BaseModel):
        email: NameEmail

    user = User(email='Fred Bloggs <fred.bloggs@example.com>')
    print(user.email)
    #> Fred Bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> Fred Bloggs

    user = User(email='fred.bloggs@example.com')
    print(user.email)
    #> fred.bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> fred.bloggs
    ```
    """  # noqa: D212

    __slots__ = 'name', 'email'

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source: type[Any],
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        import_email_validator()

        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.json_or_python_schema(
                json_schema=core_schema.str_schema(),
                python_schema=core_schema.union_schema(
                    [core_schema.is_instance_schema(cls), core_schema.str_schema()],
                    custom_error_type='name_email_type',
                    custom_error_message='Input is not a valid NameEmail',
                ),
                serialization=core_schema.to_string_ser_schema(),
            ),
        )

    @classmethod
    def _validate(cls, input_value: Self | str, /) -> Self:
        if isinstance(input_value, str):
            name, email = validate_email(input_value)
            return cls(name, email)
        else:
            return input_value

    def __str__(self) -> str:
        if '@' in self.name:
            return f'"{self.name}" <{self.email}>'

        return f'{self.name} <{self.email}>'


IPvAnyAddressType: TypeAlias = 'IPv4Address | IPv6Address'
IPvAnyInterfaceType: TypeAlias = 'IPv4Interface | IPv6Interface'
IPvAnyNetworkType: TypeAlias = 'IPv4Network | IPv6Network'

if TYPE_CHECKING:
    IPvAnyAddress = IPvAnyAddressType
    IPvAnyInterface = IPvAnyInterfaceType
    IPvAnyNetwork = IPvAnyNetworkType
else:

    class IPvAnyAddress:
        """Validate an IPv4 or IPv6 address.

        ```py
        from pydantic import BaseModel
        from pydantic.networks import IPvAnyAddress

        class IpModel(BaseModel):
            ip: IPvAnyAddress

        print(IpModel(ip='127.0.0.1'))
        #> ip=IPv4Address('127.0.0.1')

        try:
            IpModel(ip='http://www.example.com')
        except ValueError as e:
            print(e.errors())
            '''
            [
                {
                    'type': 'ip_any_address',
                    'loc': ('ip',),
                    'msg': 'value is not a valid IPv4 or IPv6 address',
                    'input': 'http://www.example.com',
                }
            ]
            '''
        ```
        """

        __slots__ = ()

        def __new__(cls, value: Any) -> IPvAnyAddressType:
            """Validate an IPv4 or IPv6 address."""
            try:
                return IPv4Address(value)
            except ValueError:
                pass

            try:
                return IPv6Address(value)
            except ValueError:
                raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            field_schema = {}
            field_schema.update(type='string', format='ipvanyaddress')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source: type[Any],
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._validate, serialization=core_schema.to_string_ser_schema()
            )

        @classmethod
        def _validate(cls, input_value: Any, /) -> IPvAnyAddressType:
            return cls(input_value)  # type: ignore[return-value]

    class IPvAnyInterface:
        """Validate an IPv4 or IPv6 interface."""

        __slots__ = ()

        def __new__(cls, value: NetworkType) -> IPvAnyInterfaceType:
            """Validate an IPv4 or IPv6 interface."""
            try:
                return IPv4Interface(value)
            except ValueError:
                pass

            try:
                return IPv6Interface(value)
            except ValueError:
                raise PydanticCustomError('ip_any_interface', 'value is not a valid IPv4 or IPv6 interface')

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            field_schema = {}
            field_schema.update(type='string', format='ipvanyinterface')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source: type[Any],
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._validate, serialization=core_schema.to_string_ser_schema()
            )

        @classmethod
        def _validate(cls, input_value: NetworkType, /) -> IPvAnyInterfaceType:
            return cls(input_value)  # type: ignore[return-value]

    class IPvAnyNetwork:
        """Validate an IPv4 or IPv6 network."""

        __slots__ = ()

        def __new__(cls, value: NetworkType) -> IPvAnyNetworkType:
            """Validate an IPv4 or IPv6 network."""
            # Assume IP Network is defined with a default value for `strict` argument.
            # Define your own class if you want to specify network address check strictness.
            try:
                return IPv4Network(value)
            except ValueError:
                pass

            try:
                return IPv6Network(value)
            except ValueError:
                raise PydanticCustomError('ip_any_network', 'value is not a valid IPv4 or IPv6 network')

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            field_schema = {}
            field_schema.update(type='string', format='ipvanynetwork')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source: type[Any],
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._validate, serialization=core_schema.to_string_ser_schema()
            )

        @classmethod
        def _validate(cls, input_value: NetworkType, /) -> IPvAnyNetworkType:
            return cls(input_value)  # type: ignore[return-value]


def _build_pretty_email_regex() -> re.Pattern[str]:
    name_chars = r'[\w!#$%&\'*+\-/=?^_`{|}~]'
    unquoted_name_group = rf'((?:{name_chars}+\s+)*{name_chars}+)'
    quoted_name_group = r'"((?:[^"]|\")+)"'
    email_group = r'<\s*(.+)\s*>'
    return re.compile(rf'\s*(?:{unquoted_name_group}|{quoted_name_group})?\s*{email_group}\s*')


pretty_email_regex = _build_pretty_email_regex()

MAX_EMAIL_LENGTH = 2048
"""Maximum length for an email.
A somewhat arbitrary but very generous number compared to what is allowed by most implementations.
"""


def validate_email(value: str) -> tuple[str, str]:
    """Email address validation using [email-validator](https://pypi.org/project/email-validator/).

    Note:
        Note that:

        * Raw IP address (literal) domain parts are not allowed.
        * `"John Doe <local_part@domain.com>"` style "pretty" email addresses are processed.
        * Spaces are striped from the beginning and end of addresses, but no error is raised.
    """
    if email_validator is None:
        import_email_validator()

    if len(value) > MAX_EMAIL_LENGTH:
        raise PydanticCustomError(
            'value_error',
            'value is not a valid email address: {reason}',
            {'reason': f'Length must not exceed {MAX_EMAIL_LENGTH} characters'},
        )

    m = pretty_email_regex.fullmatch(value)
    name: str | None = None
    if m:
        unquoted_name, quoted_name, value = m.groups()
        name = unquoted_name or quoted_name

    email = value.strip()

    try:
        parts = email_validator.validate_email(email, check_deliverability=False)
    except email_validator.EmailNotValidError as e:
        raise PydanticCustomError(
            'value_error', 'value is not a valid email address: {reason}', {'reason': str(e.args[0])}
        ) from e

    email = parts.normalized
    assert email is not None
    name = name or parts.local_part
    return name, email


__getattr__ = getattr_migration(__name__)






## Modelo de datos (simple)
- Evento: `id`, `titulo`, `fecha_inicio`, `fecha_fin`, `cupo`, `inscritos_count`.
- Inscripción: `evento_id`, `usuario_email`.

## Reglas de negocio
- `fecha_fin >= fecha_inicio`.
- `cupo >= 0` y no inscribir si el cupo está completo.
- No permitir inscripciones duplicadas para el mismo `usuario_email` y `evento_id`.

## Manejo de errores
- Respuestas claras con `status` y `message`.
- Casos esperados: validación (datos inválidos), no encontrado, conflicto (duplicado/cupo).

## Entregables
- Código fuente y breve instrucción de ejecución.
- Ejemplos de uso (curl o comandos) que prueben los casos principales.

## Criterios de aceptación
- Los endpoints/comandos funcionan según lo descrito.
- Validaciones y reglas de negocio aplicadas.
- Manejo de errores claro y consistente.

## Lo que se evaluará
- Claridad y estructura del código.
- Simplicidad y correctitud de la solución.
- Manejo de errores y validaciones.
- Priorización y enfoque bajo límite de tiempo.

## Ejemplos de uso (orientativos)
- Crear evento:
  - HTTP: `POST /events` body `{"titulo":"Meetup","fecha_inicio":"2025-01-10T18:00:00","fecha_fin":"2025-01-10T20:00:00","cupo":2}`
  - CLI: `events create --titulo "Meetup" --inicio 2025-01-10T18:00:00 --fin 2025-01-10T20:00:00 --cupo 2`
- Listar eventos:
  - HTTP: `GET /events`
  - CLI: `events list`
- Inscribir usuario:
  - HTTP: `POST /events/1/register` body `{"email":"user@example.com"}`
  - CLI: `events register 1 --email user@example.com`

## Extras opcionales (solo si queda tiempo)
- Filtro por texto en `GET /events`.
- Paginación simple.
- Test unitario básico de reglas (duplicados y cupo).
## Evaluación Integral de Habilidades y Preferencias

---

## SECCIÓN 1: INFORMACIÓN GENERAL Y PERFIL

### 1.1 Datos Básicos
- **Nombre Completo:**
- **Email:**
- **LinkedIn/GitHub (opcional):**
- **Años de experiencia en tecnología:**
- **Ocupación actual:** (Estudiante / Desarrollador / DevOps / Otro)

### 1.2 Nivel de Experiencia
Marca tu nivel en las siguientes áreas (1=Principiante, 5=Experto):

| Área | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|
| Frontend |  |  |  | x |  |
| Backend |  |  |  | x |  |
| DevOps/Cloud x |  |  |  |  |  |
| Mobile |  | x |  |  |  |
| Videojuegos |  | x |  |  |  |
| IA/ML | x |  |  |  |  |
| Bases de Datos |  |  |  | x |  |

---

## SECCIÓN 2: LENGUAJES DE PROGRAMACIÓN

### 2.1 Selección Múltiple
**Marca todos los lenguajes que dominas o has utilizado:**

- [ ] C#
- [x] TypeScript/JavaScript
- [x] Python
- [ ] Java
- [ ] C++
- [ ] Go
- [ ] Rust
- [x] PHP
- [ ] Ruby
- [ ] Swift/Kotlin
- [ ] Otro: _______________

### 2.2 Pregunta Práctica - Elige UNO de los siguientes desafíos según tu lenguaje preferido:

**Opción A (C#):**
```
Escribe una función que reciba una lista de números enteros y retorne 
un diccionario con la frecuencia de cada número. Ejemplo:
Input: [1, 2, 2, 3, 3, 3]
Output: {1: 1, 2: 2, 3: 3}
```

**Opción B (Python/JavaScript/TypeScript):**
```
Implementa una función que determine si una cadena es un palíndromo,
ignorando espacios y mayúsculas. Explica la complejidad temporal.
```

**Opción C (Java/C++):**
```
Crea una clase que implemente una pila (Stack) con las operaciones
push, pop, peek y isEmpty. Incluye manejo de excepciones.
```

**Opción D (Lenguaje de tu elección):**
```
Resuelve el siguiente problema: Dada una lista de números,
encuentra los dos números que suman un valor específico.
Optimiza tu solución y explica el tiempo de ejecución.
```

**Tu solución aquí:**
```
[Escribe tu código y explicación]
```
def es_palindromo(cadena: str) -> bool:
    """
    Determina si una cadena es un palíndromo,
    ignorando espacios y mayúsculas.
    """
    # 1. Normalizar: quitar espacios y convertir a minúsculas
    limpia = ''.join(c for c in cadena.lower() if c.isalnum())
    
    # 2. Comparar con su versión invertida
    return limpia == limpia[::-1]


# Ejemplos de uso
print(es_palindromo("Anita lava la tina"))       # True
print(es_palindromo("A man a plan a canal Panama"))  # True
print(es_palindromo("Hola Mundo"))               # False

---

## SECCIÓN 3: FRONTEND

### 3.1 Conocimientos
**Marca las tecnologías que conoces:**

- [ ] React
- [ ] Angular
- [ ] Vue.js
- [ ] Blazor
- [ ] Next.js
- [ ] Svelte
- [x] HTML5/CSS3
- [ ] Tailwind CSS
- [ ] Bootstrap
- [ ] Otro: _______________

### 3.2 Pregunta Conceptual
**¿Cuál es la diferencia entre Server-Side Rendering (SSR) y Client-Side Rendering (CSR)? ¿Cuándo usarías cada uno?**

```
[Tu respuesta aquí]
```

### 3.3 Ejercicio Práctico
**Describe cómo estructurarías un componente reutilizable de "Tarjeta de Usuario" que muestre: foto, nombre, email y botón de acción. Incluye el manejo de estados (loading, error, success).**

```
[Tu respuesta con código o pseudocódigo]
```

---

## SECCIÓN 4: BACKEND

### 4.1 Tecnologías Backend
**Marca tus experiencias:**

- [ ] .NET Core/ASP.NET
- [ ] Node.js (Express, NestJS, etc.)
- [ ] Django/Flask
- [ ] Spring Boot
- [ ] Ruby on Rails
- [x] PHP (Laravel, Symfony)
- [ ] Go (Gin, Echo)
- [ ] Otro: _______________

### 4.2 Arquitectura
**Pregunta:** Explica las diferencias entre arquitectura monolítica, microservicios y serverless. ¿Cuándo recomendarías cada una?

```
[Tu respuesta aquí]
```

### 4.3 APIs y Comunicación
**Diseña una API RESTful para un sistema de gestión de biblioteca con las siguientes entidades:**
- Libros (título, autor, ISBN, disponibilidad)
- Usuarios (nombre, email, libros prestados)
- Préstamos (usuario, libro, fecha préstamo, fecha devolución)

**Define los endpoints principales (método HTTP, ruta, descripción):**

```
[Tu respuesta aquí]
```

### 4.4 Bases de Datos
**Marca tus conocimientos:**

- [x] SQL Server
- [x] PostgreSQL
- [ ] MySQL
- [x] MongoDB
- [ ] Redis
- [ ] CosmosDB
- [ ] Cassandra
- [ ] Otro: _______________

**Pregunta:** ¿Cuándo elegirías una base de datos SQL vs NoSQL? Proporciona ejemplos concretos.

```
[Tu respuesta aquí]
```

---

## SECCIÓN 5: DEVOPS Y CLOUD

### 5.1 Experiencia en Cloud
**Marca las plataformas que has utilizado:**

- [x] Microsoft Azure
- [ ] AWS
- [ ] Google Cloud Platform
- [ ] Ninguna (aún)

**Servicios específicos que conoces:**
- [ ] Azure App Service / AWS EC2
- [ ] Azure Functions / AWS Lambda
- [ ] Azure DevOps / GitHub Actions
- [x] Docker
- [ ] Kubernetes
- [ ] Terraform / ARM Templates
- [ ] CI/CD Pipelines
- [ ] Otro: _______________

### 5.2 Pregunta Práctica
**Describe el flujo de un pipeline CI/CD básico para una aplicación web. Incluye: build, tests, deployment.**

```
[Tu respuesta aquí]
```

### 5.3 Contenedores
**¿Cuál es la diferencia entre una imagen Docker y un contenedor? ¿Cómo crearías una imagen para una aplicación Node.js simple?**

```
[Tu respuesta aquí]
```

---

## SECCIÓN 6: VIDEOJUEGOS

### 6.1 Experiencia en Game Development
**Nivel de experiencia:**
- [ ] Nunca he desarrollado videojuegos
- [ ] He hecho proyectos personales/tutoriales
- [ ] He participado en game jams
- [ ] Trabajo profesionalmente en la industria

### 6.2 Engines y Tecnologías
**Marca los que conoces:**

- [ ] Unity (C#)
- [ ] Unreal Engine (C++, Blueprints)
- [ ] Godot
- [ ] GameMaker
- [ ] HTML5 Canvas/WebGL
- [ ] Three.js / Babylon.js
- [ ] Otro: _______________

### 6.3 Conceptos de Game Development
**Responde brevemente:**

1. **¿Qué es un Game Loop y cuáles son sus componentes principales?**
```
[Tu respuesta]
```

2. **Explica la diferencia entre físicas 2D y 3D en videojuegos:**
```
[Tu respuesta]
```

3. **¿Qué patrones de diseño conoces que sean comunes en videojuegos? (ej: Singleton, Observer, State Machine)**
```
[Tu respuesta]
```

### 6.4 Desafío de Game Design
**Diseña en pseudocódigo o lenguaje de tu elección un sistema simple de inventario para un RPG que permita:**
- Agregar items
- Remover items
- Verificar si hay espacio
- Limitar el peso total

```
[Tu solución aquí]
```

---

## SECCIÓN 7: INTELIGENCIA ARTIFICIAL Y MACHINE LEARNING

### 7.1 Experiencia con IA
- [ ] No tengo experiencia
- [x] Conceptos básicos/teóricos
- [ ] He implementado modelos
- [ ] Trabajo profesionalmente con IA

### 7.2 Tecnologías
**Marca las que conoces:**

- [x] Azure AI Services
- [x] OpenAI API
- [ ] TensorFlow
- [ ] PyTorch
- [ ] Scikit-learn
- [ ] LangChain
- [ ] Hugging Face
- [ ] Otro: _______________

### 7.3 Pregunta Conceptual
**Explica brevemente qué es RAG (Retrieval-Augmented Generation) y en qué casos lo usarías:**

```
[Tu respuesta aquí]
```

---

## SECCIÓN 8: SEGURIDAD

### 8.1 Conceptos Básicos
**Explica brevemente cómo prevenirías los siguientes ataques:**

1. **SQL Injection:**
```
[Tu respuesta]
```

2. **XSS (Cross-Site Scripting):**
```
[Tu respuesta]
```

3. **CSRF (Cross-Site Request Forgery):**
```
[Tu respuesta]
```

### 8.2 Autenticación
**¿Cuál es la diferencia entre autenticación y autorización? ¿Qué es JWT y cómo funciona?**

```
[Tu respuesta aquí]
```

---

## SECCIÓN 9: METODOLOGÍAS Y BUENAS PRÁCTICAS

### 9.1 Desarrollo de Software
**Marca las metodologías que has utilizado:**

- [x] Scrum
- [ ] Kanban
- [ ] XP (Extreme Programming)
- [ ] Waterfall
- [ ] Otra: _______________

### 9.2 Control de Versiones
**Experiencia con Git:**
- [x] Básico (commit, push, pull)
- [ ] Intermedio (branches, merge, rebase)
- [ ] Avanzado (workflows, resolución de conflictos complejos)

**Pregunta:** Explica el flujo de trabajo de Git Flow o GitHub Flow:

```
[Tu respuesta aquí]
```

### 9.3 Testing
**Marca los tipos de testing que has implementado:**

- [ ] Unit Testing
- [ ] Integration Testing
- [ ] E2E Testing
- [ ] TDD (Test-Driven Development)
- [ ] BDD (Behavior-Driven Development)

**¿Qué frameworks de testing conoces?**
```
[Tu respuesta]
```

---

## SECCIÓN 10: PREFERENCIAS Y DIRECCIÓN DE LA COMUNIDAD

### 10.1 Intereses Personales
**Ordena por prioridad (1=mayor interés, 10=menor interés) los siguientes temas:**

- [x] Desarrollo Web (Frontend/Backend)
- [ ] DevOps y Cloud Computing
- [ ] Desarrollo de Videojuegos
- [x] Inteligencia Artificial / Machine Learning
- [x] Desarrollo Mobile
- [x] Ciberseguridad
- [ ] IoT (Internet of Things)
- [ ] Blockchain / Web3
- [ ] Data Science / Big Data
- [ ] AR/VR (Realidad Aumentada/Virtual)

### 10.2 Formato de Eventos
**¿Qué tipo de eventos te gustaría que organizara la comunidad?**

- [ ] Workshops técnicos presenciales
- [ ] Webinars online
- [x] Hackathons
- [ ] Game Jams
- [ ] Charlas inspiracionales
- [ ] Sesiones de pair programming
- [x] Code reviews grupales
- [ ] Networking events
- [ ] Certificaciones guiadas
- [ ] Otro: _______________

### 10.3 Nivel de Contenido
**Prefieres contenido:**
- [ ] Principiante (conceptos básicos)
- [x] Intermedio (aplicaciones prácticas)
- [x] Avanzado (arquitecturas complejas, optimización)
- [ ] Mixto (diversidad de niveles)

### 10.4 Tecnologías Microsoft
**¿Qué productos/servicios de Microsoft te interesan más?**

- [ ] Azure (Cloud Computing)
- [ ] .NET / C#
- [x] Visual Studio / VS Code
- [x] GitHub / GitHub Copilot
- [ ] Power Platform (Power Apps, Power Automate)
- [x] Microsoft 365 Development
- [ ] Xbox Game Development
- [ ] Microsoft AI (Azure OpenAI, Cognitive Services)
- [ ] Dynamics 365
- [ ] Otro: _______________

### 10.5 Colaboración
**¿Cómo te gustaría contribuir a la comunidad?**

- [x] Asistir a eventos como participante
- [x] Dar charlas/talleres
- [x] Mentorear a otros miembros
- [x] Contribuir con contenido (blogs, tutoriales)
- [ ] Organizar eventos
- [x] Ayudar en proyectos open source de la comunidad
- [ ] Otro: _______________

### 10.6 Horarios
**¿Qué horarios te vienen mejor para eventos online?**
- [ ] Mañana (8am - 12pm)
- [x] Tarde (12pm - 6pm)
- [ ] Noche (6pm - 10pm)
- [x] Fines de semana

### 10.7 Proyectos Comunitarios
**¿Te interesaría participar en proyectos colaborativos de la comunidad?**
- [ ] Sí, en proyectos open source
- [x] Sí, en proyectos de aprendizaje
- [x] Sí, en competencias/hackathons
- [ ] Por ahora solo quiero aprender
- [ ] Otro: _______________

### 10.8 Visión de la Comunidad
**En tu opinión, ¿cuál debería ser el enfoque principal de la comunidad Microsoft? (respuesta abierta)**

```
Bueno, para mí es algo nuevo, la visión que tengo para esta comunidad es principalmente trabajar de manera colavorativa, proactiva, y con respecto al enfoque, driría que se podría organizar eventos, pues así podríamos a motivar a los estudiantes a impulsarse como profesionales manteniendo competencias de sus conocemientos con otros alumnos.
```

### 10.9 Desafíos Actuales
**¿Cuáles son los mayores desafíos técnicos que enfrentas actualmente en tu desarrollo profesional?**

```
[Podría ser dependencia de la IAs]

```

### 10.10 Sugerencias Adicionales
**¿Qué más te gustaría ver en la comunidad? (recursos, beneficios, eventos especiales, etc.)**

```
[Tus ideas y sugerencias aquí]
```

---

## SECCIÓN 11: CASO PRÁCTICO INTEGRAL (OPCIONAL - PERO RECOMENDADO)

### Proyecto de Evaluación Completa
**Elige UNO de los siguientes proyectos y describe tu enfoque (no necesitas implementarlo, solo diseñarlo):**

#### Opción A: Sistema de E-Learning
Diseña la arquitectura de una plataforma educativa que incluya:
- Frontend para estudiantes y profesores
- Backend con API REST
- Sistema de videollamadas
- Almacenamiento de videos
- Gamificación con puntos y logros
- Despliegue en la nube

**Describe:**
1. Stack tecnológico elegido y por qué
2. Arquitectura general (diagrama o descripción)
3. Principales desafíos técnicos
4. Estrategia de escalabilidad
5. Consideraciones de seguridad

#### Opción B: Juego Multiplayer
Diseña un juego simple online multiplayer:
- Engine/framework a utilizar
- Arquitectura cliente-servidor
- Sincronización de estado
- Manejo de latencia
- Sistema de matchmaking básico
- Persistencia de datos de jugadores

**Describe:**
1. Tecnologías elegidas
2. Flujo de conexión y sincronización
3. Estructura de datos del juego
4. Desafíos de networking
5. Plan de monetización (opcional)

#### Opción C: Sistema de IA Empresarial
Diseña un asistente de IA para una empresa que:
- Responda preguntas sobre documentos internos (RAG)
- Se integre con Microsoft Teams
- Tenga memoria de conversaciones
- Incluya métricas y analytics
- Sea seguro y cumpla con privacidad

**Describe:**
1. Arquitectura de la solución
2. Servicios de Azure/IA a utilizar
3. Pipeline de procesamiento de documentos
4. Estrategia de seguridad y compliance
5. KPIs para medir efectividad

**Tu diseño aquí:**
```
[Tu propuesta completa - puede incluir diagramas en texto, arquitectura, pseudocódigo, etc.]
```

---

## SECCIÓN FINAL: REFLEXIÓN

### ¿Por qué quieres unirte a esta comunidad?
```
[Me gustaría aprender cosas nuevas y trabajar de manera colavorativa]
```

### ¿Qué esperas aprender o lograr en los próximos 6 meses?
```
[Tu respuesta]
```

### ¿Algún comentario adicional?
```
[Espacio libre para cualquier cosa que quieras compartir]
```

---

## INSTRUCCIONES DE ENTREGA

- **Formato:** Puedes responder en este mismo documento o crear un repositorio GitHub con tus respuestas
- **Tiempo estimado:** No hay límite, pero recomendamos 2-4 horas
- **Código:** Si incluyes código, asegúrate de que sea legible y comentado
- **Honestidad:** No hay respuestas incorrectas en las secciones de opinión. Queremos conocer tu perspectiva real

---

## EVALUACIÓN

Esta prueba nos ayudará a:
1. ✅ Entender tu nivel técnico en diferentes áreas
2. ✅ Conocer tus intereses y hacia dónde quieres crecer
3. ✅ Diseñar contenido relevante para la comunidad
4. ✅ Conectarte con otros miembros con intereses similares
5. ✅ Identificar potenciales speakers y mentores

**¡No es una prueba de eliminación, es una herramienta de conexión!**

---

**Gracias por tomarte el tiempo de completar esta evaluación.**
**¡Bienvenido a la Comunidad Microsoft! 🚀**

---

*Versión 1.0 - Comunidad Microsoft*

*Para consultas: [mmoya0992@gmail.com]*
