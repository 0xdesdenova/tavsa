from django.http import StreamingHttpResponse
from rest_framework import views, status, response
from genetic_algorithm import solution


# Create your views here.
def solve(request):
    yield 'Started'
    yield solution.main(request.data['num_of_vehicles'], request.data['atm_list'], request.data['date'])
    yield 'Finished'


class Solve(views.APIView):
    def post(self, request):
        return StreamingHttpResponse(solve(request))