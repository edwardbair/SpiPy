{% extends "automodsumm/summary.rst" %}

{% block object_members %}
.. automethod:: {{ fullname }}.{{ obj_name }}
{% endblock %}