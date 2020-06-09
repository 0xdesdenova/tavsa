from django.http import StreamingHttpResponse
from rest_framework import views, status, response
from genetic_algorithm import solution


# Create your views here.
def solve(request):
    yield 'Started'
    yield solution.main(request.data['date'], request.data['population'], request.data['generations'], request.data['num_of_vehicles'], request.data['atm_list'], request.data['human_routes'])
    yield 'Finished'


class Solve(views.APIView):
    def post(self, request):
        print(len(request.data['atm_list']))
        sum = 0
        for atms in request.data['human_routes']:
            sum += atms
        print(sum)
        return StreamingHttpResponse(solve(request))