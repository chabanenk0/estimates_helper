
(function($) {
    $(function () {
        $('.task-checkbox').on('click', function(event) {
            var taskIds = []
            $('.task-checkbox:checked').each(function (i, item) {
                taskIds.push($(item).data('id'));
            });
            $('#tasks-selected').val(taskIds.join(','));
        });

        $('#checkbox-select-all').on('change', function (event) {
            var state =$(event.currentTarget).prop('checked');
            $('.task-checkbox').prop('checked', state);
            $('.task-checkbox').first().trigger('click').trigger('click');
        })
    })
})(jQuery)